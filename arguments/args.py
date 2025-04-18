from argparse import ArgumentParser
import yaml
from types import SimpleNamespace
import os


class Args:
    def __init__(self, default_args_path: str, args):
        self.parser = ArgumentParser()

        # open yaml file
        with open(default_args_path, "r") as f:
            self.default_args = yaml.safe_load(f)
        for class_level in self.default_args:
            for arg_level in self.default_args[class_level]:
                self.parser.add_argument(
                    f"--{arg_level}",
                    type=type(self.default_args[class_level][arg_level]),
                    required=False,
                )
        self.parser.add_argument(
            "--config", type=str, help="Path to the config file", required=False
        )
        self.args = self.parser.parse_args(args)

        if hasattr(self.args, "config") and self.args.config is not None:
            config_path = self.args.config
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
        else:
            config = {}

        self.arg_dict_ = {}
        for class_level in self.default_args:
            class_level_dict = {}
            for arg_level in self.default_args[class_level]:
                if (
                    hasattr(self.args, arg_level)
                    and getattr(self.args, arg_level) is not None
                ):
                    class_level_dict[arg_level] = getattr(self.args, arg_level)
                elif class_level in config and arg_level in config[class_level]:
                    class_level_dict[arg_level] = config[class_level][arg_level]
                else:
                    class_level_dict[arg_level] = self.default_args[class_level][
                        arg_level
                    ]
            self.arg_dict_[class_level] = class_level_dict

            if not hasattr(self, class_level):
                setattr(
                    self,
                    class_level,
                    SimpleNamespace(**class_level_dict, ka=class_level_dict),
                )

        # print(
        #     [
        #         f"{class_level}: {class_level_dict}"
        #         for class_level, class_level_dict["ka"] in self.arg_dict_.items()
        #     ]
        # )

    def __getitem__(self, item):
        return self.arg_dict_[item]

    def save(self):
        pth = os.path.join(
            self.logging.save_dir,
            self.logging.name,
            self.logging.version,
            "config.yaml",
        )
        os.makedirs(os.path.dirname(pth), exist_ok=True)
        with open(
            pth,
            "w",
        ) as f:
            yaml.dump(self.arg_dict_, f)


if __name__ == "__main__":
    args = Args("./arguments/default_args.yaml", [])

    # three ways to access an element in args
    print(args["main"]["codebook_size"])
    print(args.main.ka["codebook_size"])
    print(args.main.codebook_size)
