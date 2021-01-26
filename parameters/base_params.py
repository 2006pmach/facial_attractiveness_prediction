class BaseParameters():
    def __init__(self):
        pass

    def __str__(self):
        msg = []
        for key, val in self.__dict__.items():
            msg.append("{}: {}".format(key, val))

        return "\n".join(msg)
