import os
import pickle
import shutil

def delete_user_from_dataset(user_name, dataset_folder="dataset"):
    person_folder = os.path.join(dataset_folder, user_name)
    if os.path.exists(person_folder):
        shutil.rmtree(person_folder)
        print(f"Deleted folder and images for user '{user_name}'.")
    else:
        print(f"No folder found for user '{user_name}' in dataset.")

def delete_user_from_encodings(user_name, encodings_file="encodings.pickle"):
    if not os.path.exists(encodings_file):
        print("No encodings.pickle file found.")
        return

    with open(encodings_file, "rb") as f:
        data = pickle.load(f)

    encodings = data["encodings"]
    names = data["names"]

    # Filter out the user's encodings
    new_encodings = []
    new_names = []
    removed_count = 0
    for encoding, name in zip(encodings, names):
        if name != user_name:
            new_encodings.append(encoding)
            new_names.append(name)
        else:
            removed_count += 1

    if removed_count == 0:
        print(f"No encodings found for user '{user_name}'.")
    else:
        print(f"Removed {removed_count} encodings for user '{user_name}'.")

    # Save the updated encodings
    with open(encodings_file, "wb") as f:
        pickle.dump({"encodings": new_encodings, "names": new_names}, f)

if __name__ == "__main__":
    user = input("Enter the name of the user to delete: ").strip()
    delete_user_from_dataset(user)
    delete_user_from_encodings(user)
    print("Done.")
