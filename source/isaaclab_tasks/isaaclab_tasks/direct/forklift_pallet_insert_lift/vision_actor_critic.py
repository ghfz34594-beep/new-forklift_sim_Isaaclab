from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.networks import MLP, EmpiricalNormalization
from rsl_rl.utils import resolve_nn_activation

from .vision_backbone import MobileNetVisionBackbone, freeze_module, load_backbone_checkpoint

LOG_STD_MIN = math.log(0.05)
LOG_STD_MAX = math.log(1.5)


class VisionActorCritic(nn.Module):
    """视觉 actor + 低维 critic。

    约定输入结构：
    - single camera: obs["policy"]["image"] : (N, 3, H, W)
    - dual camera:   obs["policy"]["image_left/right"] : (N, 3, H, W)
    - obs["policy"]["proprio"] : (N, P)
    - obs["critic"]            : (N, C)
    """

    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        backbone_type: str = "mobilenet_v3_small",
        pretrained_backbone_path: str | None = None,
        freeze_backbone: bool = False,
        freeze_backbone_updates: int = 0,
        imagenet_backbone_init: bool = False,
        dual_camera: bool | None = None,
        squash_actor_mean: bool = False,
        actor_action_scale: float | list[float] | tuple[float, ...] | None = None,
        **kwargs,
    ):
        if kwargs:
            print(
                "VisionActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        self.obs_groups = obs_groups
        self.noise_std_type = noise_std_type
        self.num_actions = int(num_actions)
        self.squash_actor_mean = bool(squash_actor_mean)

        sample_images, sample_proprio = self._extract_policy_terms(obs)
        sample_critic = obs["critic"]
        self.dual_camera = bool(dual_camera) if dual_camera is not None else len(sample_images) == 2
        action_scale = self._make_action_scale(actor_action_scale, self.num_actions)
        self.register_buffer("actor_action_scale", action_scale.view(1, -1), persistent=False)

        proprio_dim = int(sample_proprio.shape[-1])
        critic_dim = int(sample_critic.shape[-1])

        self.pretrained_backbone_path = pretrained_backbone_path
        self.freeze_backbone_updates = int(freeze_backbone_updates or 0)
        self._backbone_is_frozen = False
        self._policy_iteration_count = 0

        if backbone_type == "mobilenet_v3_small":
            from .vision_backbone import MobileNetVisionBackbone
            self.image_encoder = MobileNetVisionBackbone(imagenet_init=bool(imagenet_backbone_init))
        elif backbone_type == "resnet18":
            from .vision_backbone import ResNet18VisionBackbone
            self.image_encoder = ResNet18VisionBackbone(imagenet_init=bool(imagenet_backbone_init))
        elif backbone_type == "resnet34":
            from .vision_backbone import ResNet34VisionBackbone
            self.image_encoder = ResNet34VisionBackbone(imagenet_init=bool(imagenet_backbone_init))
        else:
            raise ValueError(f"Unknown backbone_type: {backbone_type}")

        with torch.no_grad():
            image_feat_dim = int(self.image_encoder(sample_images[0][:1].detach().cpu()).shape[-1])
        fused_image_feat_dim = image_feat_dim * (2 if self.dual_camera else 1)

        self.image_proj = nn.Sequential(
            nn.Linear(fused_image_feat_dim, 256),
            resolve_nn_activation(activation),
            nn.Linear(256, 256),
            resolve_nn_activation(activation),
        )
        self.proprio_encoder = MLP(proprio_dim, 128, [128, 128], activation)
        self.actor = MLP(256 + 128, num_actions, actor_hidden_dims, activation)
        self.critic = MLP(critic_dim, 1, critic_hidden_dims, activation)

        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(proprio_dim)
        else:
            self.actor_obs_normalizer = nn.Identity()

        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(critic_dim)
        else:
            self.critic_obs_normalizer = nn.Identity()

        if self.pretrained_backbone_path:
            missing, unexpected = load_backbone_checkpoint(self.image_encoder, self.pretrained_backbone_path)
            print(
                f"Loaded pretrained backbone from {self.pretrained_backbone_path}; "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )
        if freeze_backbone or self.freeze_backbone_updates > 0:
            self._set_backbone_frozen(True)
            if self.freeze_backbone_updates > 0:
                print(
                    "Vision backbone freeze schedule: "
                    f"frozen for the first {self.freeze_backbone_updates} policy iterations"
                )
            else:
                print("Vision backbone freeze schedule: frozen for the entire training run")

        print(f"Vision image encoder: {self.image_encoder}")
        print(f"Vision image projection: {self.image_proj}")
        print(f"Vision proprio encoder: {self.proprio_encoder}")
        print(f"Actor head: {self.actor}")
        if self.squash_actor_mean:
            print(f"Actor mean bounded with tanh and action scale={self.actor_action_scale.flatten().tolist()}")
        print(f"Critic MLP: {self.critic}")

        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution = None
        Normal.set_default_validate_args(False)

    @staticmethod
    def _make_action_scale(
        actor_action_scale: float | list[float] | tuple[float, ...] | None,
        num_actions: int,
    ) -> torch.Tensor:
        if actor_action_scale is None:
            scale = torch.ones(num_actions, dtype=torch.float32)
        elif isinstance(actor_action_scale, (float, int)):
            scale = torch.full((num_actions,), float(actor_action_scale), dtype=torch.float32)
        else:
            scale = torch.tensor(list(actor_action_scale), dtype=torch.float32).flatten()
            if scale.numel() == 1:
                scale = scale.repeat(num_actions)
        if int(scale.numel()) != int(num_actions):
            raise ValueError(
                f"actor_action_scale must be scalar or length {num_actions}, got shape={tuple(scale.shape)}"
            )
        if torch.any(scale <= 0.0):
            raise ValueError(f"actor_action_scale values must be positive, got {scale.tolist()}")
        return scale

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def _ensure_image_tensor(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4:
            raise ValueError(f"Expected image tensor with 4 dims, got shape={tuple(image.shape)}")
        if image.shape[1] == 3:
            return image.float()
        if image.shape[-1] == 3:
            return image.permute(0, 3, 1, 2).float()
        raise ValueError(f"Unexpected image shape={tuple(image.shape)}")

    def _extract_policy_terms(self, obs):
        if "image_left" in obs.keys() and "image_right" in obs.keys() and "proprio" in obs.keys():
            return (obs["image_left"], obs["image_right"]), obs["proprio"]
        if "image" in obs.keys() and "proprio" in obs.keys():
            return (obs["image"],), obs["proprio"]

        policy_obs = obs["policy"]
        if (
            hasattr(policy_obs, "keys")
            and "image_left" in policy_obs.keys()
            and "image_right" in policy_obs.keys()
            and "proprio" in policy_obs.keys()
        ):
            return (policy_obs["image_left"], policy_obs["image_right"]), policy_obs["proprio"]
        if hasattr(policy_obs, "keys") and "image" in policy_obs.keys() and "proprio" in policy_obs.keys():
            return (policy_obs["image"],), policy_obs["proprio"]

        raise KeyError("Cannot resolve policy image/proprio or image_left/image_right/proprio from observations")

    def _encode_policy_obs(self, obs):
        images, proprio = self._extract_policy_terms(obs)
        if self.dual_camera and len(images) != 2:
            raise ValueError("VisionActorCritic configured for dual_camera=True but observations do not contain two images")
        if not self.dual_camera and len(images) != 1:
            raise ValueError("VisionActorCritic configured for single camera but observations contain two images")
        encoded_images = []
        for image in images:
            image = self._ensure_image_tensor(image)
            if image.max() > 1.0:
                image = image / 255.0
            image = torch.clamp(image, 0.0, 1.0)
            encoded_images.append(self.image_encoder(image))

        proprio = proprio.float()
        proprio = self.actor_obs_normalizer(proprio)

        image_feat = self.image_proj(torch.cat(encoded_images, dim=-1))
        proprio_feat = self.proprio_encoder(proprio)
        return torch.cat([image_feat, proprio_feat], dim=-1)

    def _actor_mean(self, obs):
        mean = self.actor(self._encode_policy_obs(obs))
        if self.squash_actor_mean:
            return torch.tanh(mean) * self.actor_action_scale.to(device=mean.device, dtype=mean.dtype)
        return mean

    def _set_backbone_frozen(self, frozen: bool) -> None:
        freeze_module(self.image_encoder, frozen)
        self._backbone_is_frozen = frozen
        state = "frozen" if frozen else "trainable"
        print(f"Vision backbone is now {state}")

    def on_policy_iteration_end(self, policy_iteration: int) -> None:
        """Track outer PPO iterations and unfreeze backbone on schedule.

        The previous implementation counted calls to ``update_normalization()``,
        which happens multiple times inside one PPO iteration. Here we count the
        outer policy updates instead so ``freeze_backbone_updates`` truly means
        "freeze for N training iterations".
        """
        self._policy_iteration_count = int(policy_iteration)
        if not self._backbone_is_frozen or self.freeze_backbone_updates <= 0:
            return
        if self._policy_iteration_count >= self.freeze_backbone_updates:
            self._set_backbone_frozen(False)
            print(
                "Vision backbone unfrozen after "
                f"{self._policy_iteration_count} completed policy iterations; "
                "becomes trainable from the next outer learning iteration "
                f"(zero-based index {self._policy_iteration_count})"
            )

    def update_distribution(self, obs):
        mean = self._actor_mean(obs)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            clamped_log_std = torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
            std = torch.exp(clamped_log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution = Normal(mean, std)

    def act(self, obs, **kwargs):
        self.update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs):
        return self._actor_mean(obs)

    def evaluate(self, obs, **kwargs):
        critic_obs = self.get_critic_obs(obs)
        critic_obs = self.critic_obs_normalizer(critic_obs)
        return self.critic(critic_obs)

    def get_actor_obs(self, obs):
        return self._encode_policy_obs(obs)

    def get_critic_obs(self, obs):
        return obs["critic"].float()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs):
        if self.actor_obs_normalization:
            _, proprio = self._extract_policy_terms(obs)
            self.actor_obs_normalizer.update(proprio.float())
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self.get_critic_obs(obs))

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True
