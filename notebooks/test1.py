def main():
    print("i'm test1")
    print_tre_volte('working1')
    with open("test2.py") as f:
        exec(f.read())

def print_tre_volte(word):
    print(word)
    print(word)
    print(word)

if __name__ == "__main__":
    print('main of test1 is being called')
    main()
