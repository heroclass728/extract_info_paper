class ManageDatabase:

    def __init__(self, database):

        self.cursor = database.cursor()
        self.db = database

    def insert_data(self, roll_num, total_marks, exam_type):

        # self.cursor.execute("CREATE TABLE IF NOT EXISTS mark_table (id INT AUTO_INCREMENT PRIMARY KEY, "
        #                     "roll_number VARCHAR(255), exam_type VARCHAR (255), total_marks VARCHAR(255))")
        self.cursor.execute("CREATE TABLE IF NOT EXISTS mark_table (id INT AUTO_INCREMENT PRIMARY KEY, "
                            "roll_number VARCHAR(255), exam_type VARCHAR (255), total_marks VARCHAR(255))")

        insert_sql = "INSERT INTO mark_table (roll_number, exam_type, total_marks) VALUES (%s, %s, %s)"
        insert_val = (roll_num, exam_type, total_marks)

        self.cursor.execute(insert_sql, insert_val)
        self.db.commit()

        print(self.cursor.rowcount, "record inserted")

    def read_data(self):

        self.cursor.execute("SELECT * FROM mark_table")
        query_result = self.cursor.fetchall()

        return query_result

    def update_data(self, field, value, id_index):

        self.cursor.execute("SELECT id FROM mark_table")
        query_result = self.cursor.fetchall()
        id_value = query_result[id_index]

        update_sql = "UPDATE mark_table SET " + field + " = " + "'{}'".format(value) + " WHERE id = " + str(id_value[0])
        self.cursor.execute(update_sql)

        self.db.commit()

        print("{}: {} successfully updated".format(field, value))

    def delete_data(self, user_id):

        delete_sql = "DELETE FROM mark_table WHERE id = " + str(user_id)
        self.cursor.execute(delete_sql)

        self.db.commit()

        print("User_Id:{} successfully deleted".format(user_id))
