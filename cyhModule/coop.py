
import torch
import torch.nn as nn
import open_clip

import argparse



class PromptLearner(nn.Module):
    def __init__(self,  classnames, clip_model,  model_name = "ViT-B-16"):
        super().__init__()
        # n_cls = len(classnames)
        # n_ctx = cfg.TRAINER.COOP.N_CTX
        # ctx_init = cfg.TRAINER.COOP.CTX_INIT
        # dtype = clip_model.dtype
        # ctx_dim = clip_model.ln_final.weight.shape[0]
        # clip_imsize = clip_model.visual.input_resolution
        # cfg_imsize = cfg.INPUT.SIZE[0]
        # assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        n_cls = len(classnames)
        n_ctx = 16  # best
        # n_ctx = 12
        ctx_init = False
        dtype = clip_model.token_embedding.weight.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.image_size
        cfg_imsize = 224
        # assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # model_name = "ViT-B-16"
        # model_name = 'ViT-B-16-plus-240'
        self.model_name = 'ViT-B-16-plus-240'
        tokenizer = open_clip.get_tokenizer(self.model_name)

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))


            prompt = tokenizer(tokenizer)

            # prompt = clip.tokenize(ctx_init)

            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # # random initialization
            # if cfg.TRAINER.COOP.CSC:
            #     print("Initializing class-specific contexts")
            #     ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            # else:
            #     print("Initializing a generic context")
            #     ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)

            # random initialization
            # print("Initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)

            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)


        # print(f'Initial context: "{prompt_prefix}"')
        # print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]

        name_lens = [len(tokenizer(name)) for name in classnames]

        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([tokenizer(p) for p in prompts])

        self.tokenized_prompts = tokenized_prompts

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        # self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION
        self.class_token_position = 'end'

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1) # [n_cls, M, D]

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts

if __name__ == "__main__":

    parser = argparse.ArgumentParser("VAND Challenge", add_help=True)
    # path
    # parser.add_argument("--train_data_path", type=str, default="../data/mvtec", help="train dataset path")
    parser.add_argument("--train_data_path", type=str, default="./data/visa", help="train dataset path")

    parser.add_argument("--save_path", type=str, default='./exps/vit_large_14_518', help='path to save results')
    parser.add_argument("--config_path", type=str, default='./open_clip/model_configs/ViT-B-16.json', help="model configs")
    # model
    #parser.add_argument("--dataset", type=str, default='mvtec', help="train dataset name")
    parser.add_argument("--dataset", type=str, default='visa', help="train dataset name")
    parser.add_argument("--model", type=str, default="ViT-B-16", help="model used")
    # parser.add_argument("--model", type=str, default="ViT-B-16-plus-240", help="model used")
    parser.add_argument("--pretrained", type=str, default="laion400m_e32", help="pretrained weight used")
    parser.add_argument("--features_list", type=int, nargs="+", default=[3, 6, 9], help="features used")
    # hyper-parameter
    parser.add_argument("--epoch", type=int, default=200, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--image_size", type=int, default=224, help="image size")
    parser.add_argument("--aug_rate", type=float, default=-1, help="image size")
    parser.add_argument("--print_freq", type=int, default=30, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=3, help="save frequency")
    args = parser.parse_args()

    class_name = ['normal object', 'abnormal object']

    model, _, preprocess = open_clip.create_model_and_transforms(args.model, args.image_size, pretrained=args.pretrained)
    test_class = PromptLearner(class_name, model)
    print(test_class.forward())





# class PromptLearner(nn.Module):
#     def __init__(self, cfg, classnames, clip_model):
#         super().__init__()
#         n_cls = len(classnames)
#         n_ctx = cfg.TRAINER.COOP.N_CTX
#         ctx_init = cfg.TRAINER.COOP.CTX_INIT
#         dtype = clip_model.dtype
#         ctx_dim = clip_model.ln_final.weight.shape[0]
#         clip_imsize = clip_model.visual.input_resolution
#         cfg_imsize = cfg.INPUT.SIZE[0]
#         assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
#
#
#         if ctx_init:
#             # use given words to initialize context vectors
#             ctx_init = ctx_init.replace("_", " ")
#             n_ctx = len(ctx_init.split(" "))
#
#             prompt = clip.tokenize(ctx_init)
#
#             with torch.no_grad():
#                 embedding = clip_model.token_embedding(prompt).type(dtype)
#             ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
#             prompt_prefix = ctx_init
#
#         else:
#             # random initialization
#             if cfg.TRAINER.COOP.CSC:
#                 print("Initializing class-specific contexts")
#                 ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
#             else:
#                 print("Initializing a generic context")
#                 ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
#             nn.init.normal_(ctx_vectors, std=0.02)
#             prompt_prefix = " ".join(["X"] * n_ctx)
#
#         print(f'Initial context: "{prompt_prefix}"')
#         print(f"Number of context words (tokens): {n_ctx}")
#
#         self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
#
#         classnames = [name.replace("_", " ") for name in classnames]
#         name_lens = [len(_tokenizer.encode(name)) for name in classnames]
#         prompts = [prompt_prefix + " " + name + "." for name in classnames]
#
#         tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
#         with torch.no_grad():
#             embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
#
#         # These token vectors will be saved when in save_model(),
#         # but they should be ignored in load_model() as we want to use
#         # those computed using the current class names
#         self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
#         self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS
#
#         self.n_cls = n_cls
#         self.n_ctx = n_ctx
#         self.tokenized_prompts = tokenized_prompts  # torch.Tensor
#         self.name_lens = name_lens
#         # self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION
#         self.class_token_position = 'end'
#
#     def forward(self):
#         ctx = self.ctx
#         if ctx.dim() == 2:
#             ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)  # [n_cls, M, D]
#
#         prefix = self.token_prefix
#         suffix = self.token_suffix
#
#         if self.class_token_position == "end":
#             prompts = torch.cat(
#                 [
#                     prefix,  # (n_cls, 1, dim)
#                     ctx,  # (n_cls, n_ctx, dim)
#                     suffix,  # (n_cls, *, dim)
#                 ],
#                 dim=1,
#             )
#
#         elif self.class_token_position == "middle":
#             half_n_ctx = self.n_ctx // 2
#             prompts = []
#             for i in range(self.n_cls):
#                 name_len = self.name_lens[i]
#                 prefix_i = prefix[i: i + 1, :, :]
#                 class_i = suffix[i: i + 1, :name_len, :]
#                 suffix_i = suffix[i: i + 1, name_len:, :]
#                 ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
#                 ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
#                 prompt = torch.cat(
#                     [
#                         prefix_i,  # (1, 1, dim)
#                         ctx_i_half1,  # (1, n_ctx//2, dim)
#                         class_i,  # (1, name_len, dim)
#                         ctx_i_half2,  # (1, n_ctx//2, dim)
#                         suffix_i,  # (1, *, dim)
#                     ],
#                     dim=1,
#                 )
#                 prompts.append(prompt)
#             prompts = torch.cat(prompts, dim=0)
#
#         elif self.class_token_position == "front":
#             prompts = []
#             for i in range(self.n_cls):
#                 name_len = self.name_lens[i]
#                 prefix_i = prefix[i: i + 1, :, :]
#                 class_i = suffix[i: i + 1, :name_len, :]
#                 suffix_i = suffix[i: i + 1, name_len:, :]
#                 ctx_i = ctx[i: i + 1, :, :]
#                 prompt = torch.cat(
#                     [
#                         prefix_i,  # (1, 1, dim)
#                         class_i,  # (1, name_len, dim)
#                         ctx_i,  # (1, n_ctx, dim)
#                         suffix_i,  # (1, *, dim)
#                     ],
#                     dim=1,
#                 )
#                 prompts.append(prompt)
#             prompts = torch.cat(prompts, dim=0)
#
#         else:
#             raise ValueError
#
#         return prompts
#
# class CustomCLIP(nn.Module):
#     def __init__(self, cfg, classnames, clip_model):
#         super().__init__()
#         self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
#         self.tokenized_prompts = self.prompt_learner.tokenized_prompts
#         self.image_encoder = clip_model.visual
#         self.text_encoder = TextEncoder(clip_model)
#         self.logit_scale = clip_model.logit_scale
#         self.dtype = clip_model.dtype
#
#     def forward(self, image):
#         image_features = self.image_encoder(image.type(self.dtype))
#
#         prompts = self.prompt_learner()
#         tokenized_prompts = self.tokenized_prompts
#         text_features = self.text_encoder(prompts, tokenized_prompts)
#
#         image_features = image_features / image_features.norm(dim=-1, keepdim=True)
#         text_features = text_features / text_features.norm(dim=-1, keepdim=True)
#
#         logit_scale = self.logit_scale.exp()
#         logits = logit_scale * image_features @ text_features.t()
#
#         return logits

