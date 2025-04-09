from argparse import ArgumentParser
import yaml
from types import SimpleNamespace


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
                    default=self.default_args[class_level][arg_level],
                )
        self.args = self.parser.parse_args(args)

        self.arg_dict_ = {}
        for class_level in self.default_args:
            class_level_dict = {}
            for arg_level in self.default_args[class_level]:
                class_level_dict[arg_level] = getattr(self.args, arg_level)
            self.arg_dict_[class_level] = class_level_dict

            if not hasattr(self, class_level):
                setattr(
                    self,
                    class_level,
                    SimpleNamespace(**class_level_dict, ka=class_level_dict),
                )

    def __getitem__(self, item):
        return self.arg_dict_[item]


if __name__ == "__main__":
    args = Args("./arguments/default_args.yaml", [])

    # three ways to access an element in args
    print(args["main"]["codebook_size"])
    print(args.main.ka["codebook_size"])
    print(args.main.codebook_size)
