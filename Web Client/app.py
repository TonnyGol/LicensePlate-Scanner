from flask import Flask , redirect , url_for , render_template , request , session , flash
from datetime import timedelta
import socket
import os
import webbrowser

#----------------------------------------------------------
SERVER_IP = "127.0.0.1"
SERVER_PORT = 12074
Login_Request = {"Code":100, "UserName":"", "PassWord":""}
Recognition_Request = {"Code":200, "Data":""}
#----------------------------------------------------------
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_address = (SERVER_IP, SERVER_PORT)
loginMsg = Login_Request
popUpmsg = Recognition_Request
#----------------------------------------------------------
app = Flask(__name__)
app.secret_key = "LPSTAC"
app.permanent_session_lifetime = timedelta(minutes=1)
#----------------------------------------------------------
@app.route("/")
def test():
	return render_template("test.html", content=["TEST", "TEST", "TEST"])
#----------------------------------------------------------
@app.route("/login", methods=["POST", "GET"])
def login():
	if request.method == "POST":
		loginMsg["UserName"] = request.form["usrname"]
		loginMsg["PassWord"] = request.form["psw"]
		msg_build = str(loginMsg)
		print (msg_build)
		sock.sendto(msg_build.encode(), server_address)
		server_msg, server_addr = sock.recvfrom(1024)
		server_msg = server_msg.decode()
		print(server_msg)
		if(server_msg == "OK"):
			session["username"] = request.form["usrname"]
			session["password"] = request.form["psw"]
			flash("Login Succesful!", "info")
			return redirect(url_for("user"))
		else:
			flash("Wrong username or password", "error")
			return render_template("Login.html")
	else:
		if "username" in session and "password" in session:
			return redirect(url_for("user"))
		return render_template("Login.html")
#----------------------------------------------------------
@app.route("/User", methods=["POST", "GET"])
def user():
	if "username" in session and "password" in session:
		if request.method == "POST":
			return redirect(url_for("logout"))
		popUpmsg["Data"] = "Wait for popUp message."
		msg_build = str(popUpmsg)
		sock.sendto(msg_build.encode(), server_address)
		server_msg, server_addr = sock.recvfrom(1024)
		server_msg = server_msg.decode()
		flash(server_msg)
		return render_template("popUp.html")
	else:
		return redirect(url_for("login"))
#----------------------------------------------------------
@app.route("/logout")
def logout():
	flash("You have been logged out")
	session.pop("username", None)
	session.pop("password", None)
	return redirect(url_for("login"))
#----------------------------------------------------------


if __name__ == "__main__":
	os.system("start http://127.0.0.1:5000/login")
	#webbrowser.open("http://127.0.0.1:5000/login")
	app.run(debug=False)
	


	
