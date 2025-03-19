from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--model_path', type=str, default="./checkpoints/model_epoch_29.pth",
                            help='path to model checkpoint')
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')

        self.isTrain = False
        return parser
