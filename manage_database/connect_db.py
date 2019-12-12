import os
import json
import mysql.connector


def connect_db(cur_dir):

    credential_path = os.path.join(cur_dir, 'source/mysql_credential.json')

    json_file = open(credential_path)
    credential = json.load(json_file)
    USERNAME = credential['username']
    PASSWORD = credential['password']

    mark_db = mysql.connector.connect(

        host="localhost",
        user=USERNAME,
        password=PASSWORD

    )

    mark_cursor = mark_db.cursor()
    mark_cursor.execute("CREATE DATABASE IF NOT EXISTS mark_database")

    mark_db = mysql.connector.connect(

        host="localhost",
        user=USERNAME,
        password=PASSWORD,
        database="mark_database"
    )

    return mark_db
