import subprocess
import select

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

    def eval_expr(self, expr: str) -> str:
        """Evaluate an expression using Rival."""
        return self.send_command(f'(eval {expr})')

    def set_precision(self, precision: int):
        """Set precision in Rival."""
        self.send_command(f'(set precision {precision})')

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
