import json
import sqlite3
import requests

def update_scryfall_json():
        r = requests.get("https://api.scryfall.com/bulk-data/default_cards")
        jsonPath = '.\default-cards.json'
        open(jsonPath, 'wb').write(r.content)
        r_json = json.load(open(jsonPath, 'r', encoding='utf8'))
        download_url = r_json['download_uri']
        cards_json = requests.get(download_url)
        open(jsonPath, 'wb').write(cards_json.content)

def create_connection():
    con = None
    try:
        dbPath = '.\card_db.sqlite'
        con = sqlite3.connect(dbPath)
    except sqlite3.Error as error:
        print("Error while connecting to sqlite", error)

    return con

def create_db(con, cur):

    jsonPath = '.\default-cards.json'

    cards = json.load(open(jsonPath, 'r', encoding='utf8'))

    cur.execute("""
                    CREATE TABLE IF NOT EXISTS main.mtg
                    (id TEXT NOT NULL UNIQUE, name TEXT NOT NULL, mana_cost TEXT,
                    cmc REAL, type_line TEXT, oracle_text TEXT,
                    power TEXT, toughness TEXT, color_identity TEXT,
                    keywords TEXT, legalities TEXT, 'set' TEXT,
                    set_name TEXT, rarity TEXT, artist TEXT,
                    prices TEXT, image TEXT NOT NULL)
                    """)

    fields = ('id', 'name', 'mana_cost', 'cmc', 'type_line', 'oracle_text', 'power', 'toughness', 'color_identity', 'keywords', 'legalities', "'set'", 'set_name', 'rarity', 'artist', 'prices', 'image')

    query = "INSERT INTO main.mtg ({}) VALUES ({})".format(",".join(fields),",".join(("?",) * len(fields)))

    for card in cards:
        data = []
        for field in fields:
            if field == 'color_identity' or field == 'keywords':
                data.append(''.join(card.get(field, '')))
            elif field == 'legalities' or field == 'prices':
                data.append(json.dumps(card.get(field, '')))
            elif field == "'set'":
                data.append(card.get('set'))
            elif field == 'id':
                data.append(card.get(field, None))
            elif field == 'image':
                data.append(card.get("image_uris", {'png': ""})['png'])
            else:
                data.append(card.get(field, None))
        cur.execute(query, data)

    con.commit()


def update_db(con, cur):
    jsonPath = '.\default-cards.json'

    cards = json.load(open(jsonPath, 'r', encoding='utf8'))

    fields = ('id', 'name', 'mana_cost', 'cmc', 'type_line', 'oracle_text', 'power', 'toughness', 'color_identity', 'keywords', 'legalities', "'set'", 'set_name', 'rarity', 'artist', 'prices', 'image')

    # check if card is in sql
    query = "SELECT * from main.mtg where id = ?"

    for card in cards:
        card_id = card.get('id', None)
        cur.execute(query, (card_id,))
        sqlData = cur.fetchall()

        # if it is update price from latest scryfall download
        if len(sqlData) != 0:
            # Update price data
            edit_price = "UPDATE main.mtg SET prices = ? WHERE id = ?"
            price_data = json.dumps(card.get('prices', ''))
            cur.execute(edit_price, (price_data, card_id))

        # else add card to sql
        elif len(sqlData) == 0:
            # Add card to db
            new_entry = "INSERT INTO main.mtg ({}) VALUES ({})".format(",".join(fields),",".join(("?",) * len(fields)))
            data = []
            for field in fields:
                if field == 'color_identity' or field == 'keywords':
                    data.append(''.join(card.get(field, '')))
                elif field == 'legalities' or field == 'prices':
                    data.append(json.dumps(card.get(field, '')))
                elif field == "'set'":
                    data.append(card.get('set'))
                elif field == 'id':
                    data.append(card.get(field, None))
                elif field == 'image':
                    data.append(card.get("image_uris", {'png': ""})['png'])
                else:
                    data.append(card.get(field, None))
            cur.execute(new_entry, data)

    con.commit()




update_scryfall_json()
connection = create_connection()
cursor = connection.cursor()

table_query = "SELECT count(name) FROM sqlite_master WHERE type='table' AND name='mtg'"
cursor.execute(table_query)

if cursor.fetchone()[0]==1:
    print("Updating database...")
    update_db(connection, cursor)
else:
    print("Creating database...")
    create_db(connection, cursor)

connection.close()
