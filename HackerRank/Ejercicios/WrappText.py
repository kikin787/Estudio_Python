import textwrap

def wrap(string, max_width):
    filled_text = textwrap.fill(string, max_width)
    return filled_text

if __name__ == '__main__':
    string, max_width = input(), int(input())
    result = wrap(string, max_width)
    print(result)