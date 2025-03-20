import re
import subprocess

from typing import Any, Optional

from .interval import RealInterval

INTERVAL_REGEX = re.compile(r"\[([^\s]+),[\s*]([^\s]+)\]")

MAX_RIVAL_PRECISION = 2 ** 14

class InsufficientPrecisionError(Exception):
    """Raised when the precision is not sufficient for evaluation."""

    expr: str
    prec: int

    def __init__(self, e: str, prec: int):
        super().__init__(f"Precision {prec} is insufficient for: {e}")
        self.expr = e
        self.prec = prec

class PrecisionLimitExceeded(Exception):

    def __init__(self, msg: str):
        super().__init__(msg)

class RivalManager:
    """Wrapper around a Rival subprocess."""

    prec: int
    """Precision to use for Rival calculations"""

    logging: bool
    """Enable logging?"""

    process: Any
    """Underlying subprocess object"""

    def __init__(self, logging: bool = False, prec: int = 53):
        """Initialize and start the Racket subprocess with the Rival library."""
        self.prec = prec
        self.logging = logging
        self.process = subprocess.Popen(
            ['racket', '-l', 'rival'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def send_command(self, command: str, wait_on_output: bool = True) -> Optional[str]:
        """Send a command to the Racket subprocess and return the output."""
        if not self.process:
            raise RuntimeError("RivalManager process is not running.")

        # send command to Rival
        if self.logging:
            print(f"Send command: {command}")
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

        # exit if we do not need to wait
        if not wait_on_output:
            return None

        # read line from stdout
        return self.process.stdout.readline().strip()


    def eval_expr(self, expr: str, fun_str: Optional[str] = None) -> bool | str | RealInterval:
        """Evaluate an expression using Rival."""
        response = self.send_command(f'(eval {expr})')
        assert response is not None, 'expected a response'

        if fun_str is None:
            fun_str = expr

        if response == "#t":
            return True
        elif response == "#f":
            return False
        elif "Could not evaluate" in response:
            raise InsufficientPrecisionError(fun_str, self.prec)
        elif 'Domain error' in response:
            return 'nan'
        elif response == '+inf.bf':
            return '+inf'
        elif response == '-inf.bf':
            return '-inf'
        else:
            matches = re.match(INTERVAL_REGEX, response)
            if matches is None:
                raise ValueError(f"Could not parse interval from response: {response}")
            else:
                lo = matches.group(1)
                hi = matches.group(2)
                return RealInterval(lo, hi, self.prec + 1)

    def set_print_ival(self, flag: bool):
        self.send_command(f"(set print-ival? #{'t' if flag else 'f'})", wait_on_output=False)

    def set_precision(self, prec: int):
        """Set precision in Rival."""
        if prec > MAX_RIVAL_PRECISION:
            raise PrecisionLimitExceeded(f'Precision {prec} exceeds maximum of {MAX_RIVAL_PRECISION}')

        self.prec = prec
        self.send_command(f'(set precision {prec})', wait_on_output=False)

    def define_function(self, function_definition: str):
        """Define a new function in Rival."""
        self.send_command(f'(define {function_definition})', wait_on_output=False)

    def close(self):
        """Close the subprocess cleanly."""
        if self.process:
            self.process.stdin.close()
            self.process.terminate()
            self.process.wait()
            self.process = None
    
    def __del__(self):
        """Destructor to ensure the subprocess is cleaned up."""
        self.close()

# Example usage
if __name__ == "__main__":
    rival = RivalManager()

    # Set precision and print mode
    rival.set_print_ival(True)
    rival.set_precision(1000)

    # Evaluate an expression
    print(rival.eval_expr("(sin 1e100)"))
    print()

    # Define a function
    rival.define_function("(f x) (- (sin x) (- x (/ (pow x 3) 6)))")
    print(rival.eval_expr("f 1e-100"))
    rival.close()
