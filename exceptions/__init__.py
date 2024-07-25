class StopExecution(Exception):
    def _render_traceback_(self):
        pass

class ValueErrorApp(Exception):
    def __init__(self, message="Value error"):
        self.message = message
        print(f"\033[41m {self.__class__.__name__}: {self.message} \033[0m")
        super().__init__(self.message)

    def _render_traceback_(self):
        pass


class AssertionErrorApp(Exception):
    def __init__(self, message="Value error"):
        self.message = message
        print(f"\033[41m {self.__class__.__name__}: {self.message} \033[0m")
        super().__init__(self.message)

    def _render_traceback_(self):
        pass
