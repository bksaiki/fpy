import subprocess
import select

from titanfp.titanic.digital import Digital
from titanfp.titanic.gmpmath import mpfr_to_digital, mpfr
from fpy2.runtime.real.interval import RealInterval

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
        
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()
        
        result_lines = []
        while True:
            ready, _, _ = select.select([self.process.stdout], [], [], 1.0) # Timeout of 1 second waiting for the result
            if self.process.stdout in ready:
                line = self.process.stdout.readline().strip()
                if not line: # Subprocess ended
                    break
                result_lines.append(line)
            else:
                break  # Timeout occurred
        
        return '\n'.join(result_lines)

    def eval_expr(self, expr: str) -> bool | RealInterval:
        """Evaluate an expression using Rival."""
        response = self.send_command(f'(eval {expr})')

        if response == "#t":
            return True
        elif response == "#f":
            return False
        elif "Could not evaluate" in response:
            raise ValueError("Evaluation failed: Could not evaluate")
        else:
            # Try to evaluate as Digital
            real = self.real_to_digital(response)
            return self.digital_to_ival(real)

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

    def digital_to_ival(self, x: Digital):
        """
        Converts `x` into an interval that represents the rounding envelope of `x`,
        i.e., the tightest set of values that would round to `x` at the
        current precision of `x`.
        """
        y = x.round_new(max_p=x.p + 1)  # increase precision by 1
        prev = y.prev_float()        # next floating-point number (toward zero)
        next = y.next_float()        # previous floating-point number (away zero)
        if x.negative:
            return RealInterval(next, prev)
        else:
            return RealInterval(prev, next)

    def real_to_digital(self, x: str | int | float):
        """
        Converts `x` into a `Digital` type without loss of accuracy.
        Raises an exception if `x` cannot be represented exactly at the given precision.
        """
        print(x)
        rto_round = mpfr(x, prec=self.prec)
        rounded = mpfr_to_digital(rto_round).round_new(max_p=self.prec)
        if rounded.inexact:
            raise ValueError(f"cannot represent {x} exactly at precision {self.prec}")
        return rounded

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
