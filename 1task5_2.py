total_jumps = 0

for i in range(10):   
    total_jumps += 10
    print("You have completed", total_jumps, "jumping jacks")

    if total_jumps == 100:
        print("Congratulations! You completed the workout")
        break

    tired = input("Are you tired? (yes/y or no/n): ").lower()

    if tired == "yes" or tired == "y":
        skip = input("Do you want to skip the remaining sets? (yes/y or no/n): ").lower()
        if skip == "yes" or skip == "y":
            print("You completed a total of", total_jumps, "jumping jacks")
            break
    else:
        remaining = 100 - total_jumps
        print(remaining, "jumping jacks remaining")