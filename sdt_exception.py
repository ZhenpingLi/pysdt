class SDTException(Exception):
    """
    Base exception class for the SDT application.
    Used to handle errors during data training and post-training analysis.
    """
    def __init__(self, message: str, context: str = None):
        self.message = message
        self.context = context
        super().__init__(self.message)

    def __str__(self):
        if self.context:
            return f"[{self.context}] {self.message}"
        return self.message
