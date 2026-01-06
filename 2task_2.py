import random

words = {
    "python": "A popular programming language",
    "java": "Used for Android development",
    "html": "Used to build web pages",
    "linux": "An open-source operating system"
}

word, hint = random.choice(list(words.items()))
guessed_letters = []
attempts = 6

hangman_stages = [
    """
     ------
     |    |
     |
     |
     |
     |
    """,
    """
     ------
     |    |
     |    O
     |
     |
     |
    """,
    """
     ------
     |    |
     |    O
     |    |
     |
     |
    """,
    """
     ------
     |    |
     |    O
     |   /|
     |
     |
    """,
    """
     ------
     |    |
     |    O
     |   /|\\
     |
     |
    """,
    """
     ------
     |    |
     |    O
     |   /|\\
     |   /
     |
    """,
    """
     ------
     |    |
     |    O
     |   /|\\
     |   / \\
     |
    """
]

print(" Welcome to Hangman!")
print("Hint:", hint)

while attempts > 0:
    print(hangman_stages[6 - attempts])
    
    display_word = ""
    for letter in word:
        if letter in guessed_letters:
            display_word += letter + " "
        else:
            display_word += "_ "
    
    print("Word:", display_word.strip())

    if "_" not in display_word:
        print(" Congratulations! You guessed the word.")
        break

    guess = input("Guess a letter: ").lower()

    if guess in guessed_letters:
        print("You already guessed that letter.")
        continue

    guessed_letters.append(guess)

    if guess not in word:
        attempts -= 1
        print(" Wrong guess!")

else:
    print(hangman_stages[-1])
    print(" Game Over! The word was:", word)