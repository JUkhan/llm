from flask import Flask, jsonify, request

app = Flask(__name__)

# Sample data for users
users = [
    {"id": 1, "name": "John", "email": "john@example.com"},
    {"id": 2, "name": "Jane", "email": "jane@example.com"}
]

@app.route("/users", methods=["GET"])
def get_users():
    return jsonify({"users": users})

@app.route("/users", methods=["POST"])
def create_user():
    new_user = {
        "id": len(users) + 1,
        "name": request.json["name"],
        "email": request.json["email"]
    }
    users.append(new_user)
    return jsonify({"message": "User created successfully!"}), 201

if __name__ == "__main__":
    app.run(debug=True)