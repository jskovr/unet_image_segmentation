import sys

class Alert(Exception):
    """Exception raised for custom error in the application."""

    def __init__(self, message, proposal):
        super().__init__(message)
        self.message = message
        self.proposal = proposal

    def __str__(self):
        # Custom string representation for when the exception is printed
        return f"\nAlert: {self.message}\nProposal: {self.proposal}"

    def __repr__(self):
        # Custom representation for debugging; shows the class name and attributes
        return f"Alert(message={self.message!r}, proposal={self.proposal!r})"

# Example of raising the exception

def raise_alert(alert, proposal):
    # try:
    raise Alert(alert, proposal)
    # except Alert as e:
    #     print(e)  # This will print the custom error message without the class name
        # print(repr(e))  # This will use the __repr__ method
    sys.exit()

