import setup_knowledge_base
from chat_bot import get_response

if __name__ == "__main__":
    while True:
        user_input = input("Bạn: ")
        if user_input.lower() == "quit":
            print("Kết thúc cuộc trò chuyện. Tạm biệt!")
            break
        response = get_response(user_input)
        print(f"Trợ lý: {response}")