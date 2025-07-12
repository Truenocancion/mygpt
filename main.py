import sys
import fire
from gpt import chat


def main():
    if len(sys.argv) > 1:
        fire.Fire(chat)
    else:
        user_prompt = input("Enter a prompt: ")
        print(chat(user_prompt))


if __name__ == '__main__':
    main()
