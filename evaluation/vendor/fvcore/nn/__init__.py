class FlopCountAnalysis:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def total(self):
        return 0


def flop_count_str(*args, **kwargs):
    return ""


def flop_count(*args, **kwargs):
    return 0, {}


def parameter_count(model):
    return {"": sum(parameter.numel() for parameter in model.parameters())}
