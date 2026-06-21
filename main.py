import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    print("Hello from ai-space!")
    logging.info("This is an info message.")
    logging.info("This is another info message.")

if __name__ == "__main__":
    main()
