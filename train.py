from me5406_project.training import train_sac


if __name__ == "__main__":
    path = train_sac()
    print(f"Saved model to {path}")
