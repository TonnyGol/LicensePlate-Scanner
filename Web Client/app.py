from flask import Flask, redirect, url_for, render_template, request, session, flash, jsonify
from datetime import timedelta
import time
import socket
import os
import webbrowser

# ----------------------------------------------------------
SERVER_IP = "127.0.0.1"
SERVER_PORT = 8000
Login_Request = {"Code": 100, "UserName": "", "PassWord": ""}
Recognition_Request = {"Code": 200, "Data": ""}
# ----------------------------------------------------------
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# Using a small timeout so the status fetch doesn't hang the web client if the server isn't running yet
sock.settimeout(2.0)
server_address = (SERVER_IP, SERVER_PORT)
loginMsg = Login_Request
popUpmsg = Recognition_Request
# ----------------------------------------------------------
app = Flask(__name__)
app.secret_key = "LPSTAC"
app.permanent_session_lifetime = timedelta(minutes=10)
# ----------------------------------------------------------

@app.route("/", methods=["POST", "GET"])
@app.route("/home", methods=["POST", "GET"])
def home():
    if request.method == "POST":
        return redirect(url_for("login"))
    else:
        return render_template("home.html")

@app.route("/login", methods=["POST", "GET"])
def login():
    if request.method == "POST":
        loginMsg["UserName"] = request.form["usrname"]
        loginMsg["PassWord"] = request.form["psw"]
        msg_build = str(loginMsg)
        print(f"Sending to server: {msg_build}")
        try:
            sock.sendto(msg_build.encode(), server_address)
            server_msg, server_addr = sock.recvfrom(1024)
            server_msg = server_msg.decode()
            print(f"Server response: {server_msg}")
            
            # Allow logic to pass if server responds with "OK"
            if server_msg == "OK":
                session["username"] = request.form["usrname"]
                session["password"] = request.form["psw"]
                flash("Authentication successful.", "success")
                time.sleep(1)
                return redirect(url_for("user"))
            else:
                flash("Invalid credentials. Please attempt again.", "error")
                return render_template("Login.html")
        except socket.timeout:
            flash("Server is generally unreachable. Ensure Server+DataBase is active.", "error")
            return render_template("Login.html")
        except Exception as e:
            flash(f"Connection Error: {str(e)}", "error")
            return render_template("Login.html")
    else:
        if "username" in session and "password" in session:
            return redirect(url_for("user"))
        return render_template("Login.html")


# ----------------------------------------------------------
@app.route("/User", methods=["POST", "GET"])
def user():
    if "username" in session and "password" in session:
        return render_template("popUp.html")
    else:
        return redirect(url_for("login"))


@app.route("/api/status", methods=["GET"])
def get_status():
    if "username" not in session:
        return jsonify({"status": "unauthorized"}), 401

    popUpmsg["Data"] = "Wait for popUp message."
    msg_build = str(popUpmsg)
    try:
        sock.sendto(msg_build.encode(), server_address)
        # We set a shorter timeout specifically for polling so it fails fast
        sock.settimeout(0.5)
        server_msg, server_addr = sock.recvfrom(1024)
        server_msg = server_msg.decode()
        return jsonify({"status": server_msg})
    except socket.timeout:
        return jsonify({"status": "No connection to server"})
    except Exception as e:
        return jsonify({"status": f"Error: {str(e)}"})
    finally:
        sock.settimeout(2.0) # Reset default timeout


# ----------------------------------------------------------
@app.route("/user_logout", methods=["POST"])
@app.route("/logout")
def logout():
    flash("You have been securely logged out.", "success")
    session.pop("username", None)
    session.pop("password", None)
    return redirect(url_for("login"))

# ----------------------------------------------------------

if __name__ == "__main__":
    # Remove the automatic window opening for development convenience unless requested
    # os.system("start http://127.0.0.1:5000/home")
    app.run(debug=False, port=5000)


