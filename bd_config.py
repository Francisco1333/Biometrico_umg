import mysql.connector

def get_connection():
    return mysql.connector.connect(
        host="t3a2vh.h.filess.io",
        user="umg_biometrico_watershelf",
        password="a3d2d4109de4e69794ce06962ce1c0bc8c6c202e",
        database="umg_biometrico_watershelf",
        port=3307
    )