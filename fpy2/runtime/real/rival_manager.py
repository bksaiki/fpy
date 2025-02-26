import subprocess
import select
import re
from fpy2.runtime.real.interval import RealInterval

class InsufficientPrecisionError(Exception):
    """Raised when the precision is not sufficient for evaluation."""
    def __init__(self, evaluation, prec):
        super().__init__(f"Precision {prec} is insufficient for: {evaluation}")

class RivalManager:
    def __init__(self):
        """Initialize and start the Racket subprocess with the Rival library."""
        self.process = subprocess.Popen(
            ['racket', '-l', 'rival'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        self.prec = None
    
    def send_command(self, command: str) -> str:
        """Send a command to the Racket subprocess and return the output."""
        if not self.process:
            raise RuntimeError("RivalManager process is not running.")
        
        print("Send command: %s" % command)
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()
        
        result_lines = []
        while True:
            ready, _, _ = select.select([self.process.stdout], [], [], 0.5) # Timeout of 0.5 second waiting for the result
            if self.process.stdout in ready:
                line = self.process.stdout.readline().strip()
                if not line: # Subprocess ended
                    break
                result_lines.append(line)
            else:
                break  # Timeout occurred

        return '\n'.join(result_lines)

# 9007199254740991/9007199254740992, 18014398509481983/18014398509481984

    def eval_expr(self, expr: str) -> bool | RealInterval:
        """Evaluate an expression using Rival."""
        response = self.send_command(f'(eval {expr})')

        if response == "#t":
            return True
        elif response == "#f":
            return False
        elif "Could not evaluate" in response:
            raise InsufficientPrecisionError(expr, self.prec)
        else:
            matches = re.findall(r"[^\s\[\],]+", response)
            assert len(matches) == 2
            return RealInterval(matches[0], matches[1])
        
    def set_print_ival(self, flag: bool):
        self.send_command(f"(set print-ival? #{'t' if flag else 'f'})")

    def set_precision(self, prec: int):
        """Set precision in Rival."""
        self.prec = prec
        self.send_command(f'(set precision {prec})')

    def define_function(self, function_definition: str):
        """Define a new function in Rival."""
        self.send_command(f'(define {function_definition})')

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

    # Set precision
    rival.set_precision(1000)

    # Evaluate an expression
    print(rival.eval_expr("(sin 1e100)"))
    print()

    # Define a function
    rival.define_function("(f x) (- (sin x) (- x (/ (pow x 3) 6)))")
    print(rival.eval_expr("f 1e-100"))
    rival.close()
