import sqlite3
from sqlite3 import Error
#-----------------------------------------------------------(All Global varibales)

#-----------------------------------------------------------(All DB functions)
class SqlObject:

    def __init__(object, path):
        try:
            object.dataBase = path
            object.connection = sqlite3.connect(object.dataBase)
            object.cursor = object.connection.cursor()
        except sqlite3.Error as error:
            print("Error while connecting to sqlite", error)

    def doQuery(object, Query):
        try:
            object.cursor.execute(Query)
            object.connection.commit()
        except sqlite3.Error as error:
            print("Error while Creating sqlite table", error)

    def create_Cars_Table(object):
        try:
            object.cursor.execute("CREATE TABLE Cars (num TEXT NOT NULL)")
            object.connection.commit()
        except sqlite3.Error as error:
            print("Error while Creating sqlite table", error)

    def insertIntoCarsTable(object, numToInsert):
        try:
            object.cursor.execute("INSERT INTO Cars (num)  VALUES  (" + numToInsert + ")")
            object.connection.commit()
        except sqlite3.Error as error:
            print("Error while inserting data to sqlite table", error)

    def checkNumInCarsDB(object, dict):
        try:
           object.cursor.execute("SELECT num FROM Cars WHERE num = " + dict["Data"])
           object.connection.commit()
           num = object.cursor.fetchone()
           if num is None:
            return "The number is not in the Data Base"
           else:
            return "The number is in the Data Base: " + num[0]
        except sqlite3.Error as error:
            print("Error while checking data from sqlite table", error)
    
    def deleteNumFromCars(object, num):
        try:
            object.cursor.execute("DELETE from Cars where num = " + num + "")
            object.connection.commit()
        except sqlite3.Error as error:
            print("Error while deleting data from sqlite table", error)

    def checkAllFromCopsTable(object, dict):
        userTrue = False
        passTrue = False
        try:
            object.cursor.execute("SELECT * FROM Cops")
            object.connection.commit()
            all_items = object.cursor.fetchall()
            for x in all_items:
                for y in x:
                    if(y == dict["UserName"]):
                        userTrue = True
                    if(y == dict["PassWord"]):
                        passTrue = True
            if(userTrue and passTrue):
                return "OK"
            else:
                return "Login Bad"
        except sqlite3.Error as error:
            print("Error while deleting data from sqlite table", error)
        
        
