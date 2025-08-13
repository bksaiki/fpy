from .defs import tests, examples

def test_eval():
    for core in tests + examples:
        fn = core
        args = [1.0 for _ in range(len(core.args))]
        print(core.name, fn(*args))

if __name__ == '__main__':
    test_eval()
