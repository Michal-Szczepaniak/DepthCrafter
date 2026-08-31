from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch

from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import (
    _resize_with_antialiasing,
    StableVideoDiffusionPipelineOutput,
    StableVideoDiffusionPipeline,
    retrieve_timesteps,
)
from diffusers.utils import logging
from diffusers.utils.torch_utils import randn_tensor

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class DepthCrafterPipeline(StableVideoDiffusionPipeline):

    @torch.inference_mode()
    def encode_video(
        self,
        video: torch.Tensor,
        chunk_size: int = 14,
    ) -> torch.Tensor:
        """
        :param video: [b, c, h, w] in range [-1, 1], the b may contain multiple videos or frames
        :param chunk_size: the chunk size to encode video
        :return: image_embeddings in shape of [b, 1024]
        """

        embeddings = []
        for i in range(0, video.shape[0], chunk_size):
            # Resize batched per-chunk (was previously done once on the whole
            # unbatched `video` tensor before this loop) - a real bug fixed
            # this session: with a large enough CHUNK_SIZE upstream, that
            # single unbatched antialiasing resize (a Gaussian-blur-based
            # torch.nn.functional.pad under the hood) OOM'd on its own even
            # though every other step in this pipeline is already chunked to
            # `chunk_size`/decode_chunk_size. Purely a per-frame spatial
            # resize with no cross-frame dependency, so chunking it here
            # produces bit-identical output to the unbatched version - same
            # already-established pattern as encode_vae_video() just below.
            video_224_chunk = _resize_with_antialiasing(
                video[i : i + chunk_size].float(), (224, 224)
            )
            video_224_chunk = (video_224_chunk + 1.0) / 2.0  # [-1, 1] -> [0, 1]
            tmp = self.feature_extractor(
                images=video_224_chunk,
                do_normalize=True,
                do_center_crop=False,
                do_resize=False,
                do_rescale=False,
                return_tensors="pt",
            ).pixel_values.to(video.device, dtype=video.dtype)
            embeddings.append(self.image_encoder(tmp).image_embeds)  # [b, 1024]

        embeddings = torch.cat(embeddings, dim=0)  # [t, 1024]
        return embeddings

    @torch.inference_mode()
    def encode_vae_video(
        self,
        video: torch.Tensor,
        chunk_size: int = 14,
    ):
        """
        :param video: [b, c, h, w] in range [-1, 1], the b may contain multiple videos or frames
        :param chunk_size: the chunk size to encode video
        :return: vae latents in shape of [b, c, h, w]
        """
        video_latents = []
        for i in range(0, video.shape[0], chunk_size):
            video_latents.append(
                self.vae.encode(video[i : i + chunk_size]).latent_dist.mode()
            )
        video_latents = torch.cat(video_latents, dim=0)
        return video_latents

    @staticmethod
    def check_inputs(video, height, width):
        """
        :param video:
        :param height:
        :param width:
        :return:
        """
        if not isinstance(video, torch.Tensor) and not isinstance(video, np.ndarray):
            raise ValueError(
                f"Expected `video` to be a `torch.Tensor` or `VideoReader`, but got a {type(video)}"
            )

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
            )

    @torch.no_grad()
    def __call__(
        self,
        video: Union[np.ndarray, torch.Tensor],
        height: int = 576,
        width: int = 1024,
        num_inference_steps: int = 25,
        guidance_scale: float = 1.0,
        window_size: Optional[int] = 110,
        noise_aug_strength: float = 0.02,
        decode_chunk_size: Optional[int] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        return_dict: bool = True,
        overlap: int = 25,
        track_time: bool = False,
        carry_latents: Optional[torch.Tensor] = None,
    ):
        """
        :param video: in shape [t, h, w, c] if np.ndarray or [t, c, h, w] if torch.Tensor, in range [0, 1]
        :param height:
        :param width:
        :param num_inference_steps:
        :param guidance_scale:
        :param window_size: sliding window processing size
        :param fps:
        :param motion_bucket_id:
        :param noise_aug_strength:
        :param decode_chunk_size:
        :param generator:
        :param latents:
        :param output_type:
        :param callback_on_step_end:
        :param callback_on_step_end_tensor_inputs:
        :param return_dict:
        :param carry_latents: final `overlap` frames of `latents_all` returned by a
            previous call (via `self.last_tail_latents`), fed back in as the seed for
            this call's own `latents_all`. This makes the caller's own chunk-to-chunk
            boundary behave exactly like an ordinary internal window transition - the
            existing crossfade blend below treats it identically to any other window's
            tail, so multi-chunk streaming gets the same continuity as an internal
            window seam instead of a post-hoc scale/shift patch. `video`'s first
            `overlap` frames must be the same source frames `carry_latents` was
            computed from (re-fed as leading context), matching how internal windows
            already re-see their own overlap region.
        :return:
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        num_frames = video.shape[0]
        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else 8
        requested_overlap = overlap
        if num_frames <= window_size:
            window_size = num_frames
            overlap = 0
        stride = window_size - overlap

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(video, height, width)

        # 2. Define call parameters
        batch_size = 1
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        self._guidance_scale = guidance_scale

        # 3. Encode input video
        if isinstance(video, np.ndarray):
            video = torch.from_numpy(video.transpose(0, 3, 1, 2))
        else:
            assert isinstance(video, torch.Tensor)
        video = video.to(device=device, dtype=self.dtype)
        video = video * 2.0 - 1.0  # [0,1] -> [-1,1], in [t, c, h, w]

        if track_time:
            start_event = torch.cuda.Event(enable_timing=True)
            encode_event = torch.cuda.Event(enable_timing=True)
            denoise_event = torch.cuda.Event(enable_timing=True)
            decode_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

        video_embeddings = self.encode_video(
            video, chunk_size=decode_chunk_size
        ).unsqueeze(
            0
        )  # [1, t, 1024]
        torch.cuda.empty_cache()
        # 4. Encode input image using VAE
        noise = randn_tensor(
            video.shape, generator=generator, device=device, dtype=video.dtype
        )
        video = video + noise_aug_strength * noise  # in [t, c, h, w]

        # pdb.set_trace()
        needs_upcasting = (
            self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        )
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)

        video_latents = self.encode_vae_video(
            video.to(self.vae.dtype),
            chunk_size=decode_chunk_size,
        ).unsqueeze(
            0
        )  # [1, t, c, h, w]

        if track_time:
            encode_event.record()
            torch.cuda.synchronize()
            elapsed_time_ms = start_event.elapsed_time(encode_event)
            print(f"Elapsed time for encoding video: {elapsed_time_ms} ms")

        torch.cuda.empty_cache()

        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        # 5. Get Added Time IDs
        added_time_ids = self._get_add_time_ids(
            7,
            127,
            noise_aug_strength,
            video_embeddings.dtype,
            batch_size,
            1,
            False,
        )  # [1 or 2, 3]
        added_time_ids = added_time_ids.to(device)

        # 6. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, None, None
        )
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        # 7. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents_init = self.prepare_latents(
            batch_size,
            window_size,
            num_channels_latents,
            height,
            width,
            video_embeddings.dtype,
            device,
            generator,
            latents,
        )  # [1, t, c, h, w]
        latents_all = (
            carry_latents.to(device=device, dtype=latents_init.dtype)
            if carry_latents is not None
            else None
        )

        idx_start = 0
        if overlap > 0:
            weights = torch.linspace(0, 1, overlap, device=device)
            weights = weights.view(1, overlap, 1, 1, 1)
        else:
            weights = None

        # Separate from `weights` above (which drives window-to-window
        # blending *inside* this call's own while-loop, and is None
        # whenever this call's own windowing degenerated to a single
        # window). This blends this call's output against a caller-
        # provided `carry_latents` tail, sized by requested_overlap - needs
        # to be its own thing because the degenerate case (num_frames <=
        # window_size) zeros `overlap`/`weights` above even when
        # carry_latents was provided and still needs blending against.
        carry_weights = None
        if carry_latents is not None:
            carry_blend_len = min(requested_overlap, num_frames)
            if carry_blend_len > 0:
                carry_weights = torch.linspace(0, 1, carry_blend_len, device=device)
                carry_weights = carry_weights.view(1, carry_blend_len, 1, 1, 1)

        torch.cuda.empty_cache()

        # inference strategy for long videos
        # two main strategies: 1. noise init from previous frame, 2. segments stitching
        while idx_start < num_frames - overlap:
            idx_end = min(idx_start + window_size, num_frames)
            self.scheduler.set_timesteps(num_inference_steps, device=device)

            # 9. Denoising loop
            latents = latents_init[:, : idx_end - idx_start].clone()
            latents_init = torch.cat(
                [latents_init[:, -overlap:], latents_init[:, :stride]], dim=1
            )

            video_latents_current = video_latents[:, idx_start:idx_end]
            video_embeddings_current = video_embeddings[:, idx_start:idx_end]

            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    if latents_all is not None and i == 0:
                        # Blend against the caller's carried-over tail
                        # (carry_latents), sized by requested_overlap - NOT
                        # `overlap`, which the num_frames <= window_size
                        # degenerate case above may have zeroed for THIS
                        # call's own (now-single) internal window. Using
                        # `overlap` here for a zeroed-overlap call broke on
                        # a `-0 == 0` slicing gotcha: latents_all[:, -0:] is
                        # the FULL carried tail (not empty), while
                        # latents[:, :0] is empty - a shape mismatch that
                        # only ever showed up on a final chunk shorter than
                        # window_size. Also clamp to this window's own
                        # frame count in case it's shorter than
                        # requested_overlap too (e.g. a very short final
                        # chunk).
                        blend_len = min(requested_overlap, latents.shape[1])
                        if blend_len > 0:
                            latents[:, :blend_len] = (
                                latents_all[:, -blend_len:]
                                + latents[:, :blend_len]
                                / self.scheduler.init_noise_sigma
                                * self.scheduler.sigmas[i]
                            )

                    latent_model_input = latents  # [1, t, c, h, w]
                    latent_model_input = self.scheduler.scale_model_input(
                        latent_model_input, t
                    )  # [1, t, c, h, w]
                    latent_model_input = torch.cat(
                        [latent_model_input, video_latents_current], dim=2
                    )
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=video_embeddings_current,
                        added_time_ids=added_time_ids,
                        return_dict=False,
                    )[0]
                    # perform guidance
                    if self.do_classifier_free_guidance:
                        latent_model_input = latents
                        latent_model_input = self.scheduler.scale_model_input(
                            latent_model_input, t
                        )
                        latent_model_input = torch.cat(
                            [latent_model_input, torch.zeros_like(latent_model_input)],
                            dim=2,
                        )
                        noise_pred_uncond = self.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=torch.zeros_like(
                                video_embeddings_current
                            ),
                            added_time_ids=added_time_ids,
                            return_dict=False,
                        )[0]

                        noise_pred = noise_pred_uncond + self.guidance_scale * (
                            noise_pred - noise_pred_uncond
                        )
                    latents = self.scheduler.step(noise_pred, t, latents).prev_sample

                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(
                            self, i, t, callback_kwargs
                        )

                        latents = callback_outputs.pop("latents", latents)

                    if i == len(timesteps) - 1 or (
                        (i + 1) > num_warmup_steps
                        and (i + 1) % self.scheduler.order == 0
                    ):
                        progress_bar.update()

            if latents_all is None:
                latents_all = latents.clone()
            elif overlap > 0:
                assert weights is not None
                # latents_all[:, -overlap:] = (
                #     latents[:, :overlap] + latents_all[:, -overlap:]
                # ) / 2.0
                latents_all[:, -overlap:] = latents[
                    :, :overlap
                ] * weights + latents_all[:, -overlap:] * (1 - weights)
                latents_all = torch.cat([latents_all, latents[:, overlap:]], dim=1)
            else:
                # Degenerate single-window call (num_frames <= window_size)
                # that still received carry_latents: no internal-window
                # overlap to blend (overlap==0), but this window's start
                # still needs reconciling against the carried tail - the
                # i==0 step above already seeded the initial noise with it,
                # this blends the final denoised output the same way the
                # overlap>0 branch does (once at noise-seed time, once at
                # the end), then replaces latents_all outright since this
                # single window's output already covers the whole call -
                # there's nothing else left to concatenate.
                assert carry_weights is not None
                blend_len = carry_weights.shape[1]
                latents[:, :blend_len] = (
                    latents[:, :blend_len] * carry_weights
                    + latents_all[:, -blend_len:] * (1 - carry_weights)
                )
                latents_all = latents

            idx_start += stride

        # Exposed for the caller to feed into a following call's `carry_latents`,
        # so a chunk-to-chunk boundary crossfades in latent space exactly like an
        # internal window boundary does, instead of needing a post-hoc scale/shift
        # fit on the decoded output. None when this call's own windowing degenerated
        # to a single window (num_frames <= window_size), since there's no overlap
        # tail to hand off in that case.
        self.last_tail_latents = (
            latents_all[:, -requested_overlap:].detach().clone()
            if requested_overlap > 0
            else None
        )

        if track_time:
            denoise_event.record()
            torch.cuda.synchronize()
            elapsed_time_ms = encode_event.elapsed_time(denoise_event)
            print(f"Elapsed time for denoising video: {elapsed_time_ms} ms")

        if not output_type == "latent":
            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
            # latents_all may be longer than `num_frames` (this call's own new
            # frames) when it was seeded via `carry_latents` - use its actual
            # length rather than the input video's frame count.
            frames = self.decode_latents(latents_all, latents_all.shape[1], decode_chunk_size)

            if track_time:
                decode_event.record()
                torch.cuda.synchronize()
                elapsed_time_ms = denoise_event.elapsed_time(decode_event)
                print(f"Elapsed time for decoding video: {elapsed_time_ms} ms")

            frames = self.video_processor.postprocess_video(
                video=frames, output_type=output_type
            )
        else:
            frames = latents_all

        self.maybe_free_model_hooks()

        if not return_dict:
            return frames

        return StableVideoDiffusionPipelineOutput(frames=frames)
