# def value_of_card(card):
#     """Determine the scoring value of a card.

#     :param card: str - given card.
#     :return: int - value of a given card.  See below for values.

#     1.  'J', 'Q', or 'K' (otherwise known as "face cards") = 10
#     2.  'A' (ace card) = 1
#     3.  '2' - '10' = numerical value.
#     """
#     if card in ['J', 'Q', 'K']:
#         return 10
#     elif card == 'A':
#         return 1
#     else:
#         return int(card)
    
# def higher_card(card_one, card_two):
#     """Determine which card has a higher value in the hand.

#     :param card_one, card_two: str - cards dealt in hand.  See below for values.
#     :return: str or tuple - resulting Tuple contains both cards if they are of equal value.

#     1.  'J', 'Q', or 'K' (otherwise known as "face cards") = 10
#     2.  'A' (ace card) = 1
#     3.  '2' - '10' = numerical value.
#     """
#     value_one = value_of_card(card_one)
#     value_two = value_of_card(card_two)

#     if value_one > value_two:
#         return card_one
#     elif value_one < value_two:
#         return card_two
#     else:
#         return (card_one, card_two)

# print(higher_card('K', 'Q'))
result = 'unadressd'.strip('uned')
print(result)  # Salida: 'unadress'
result = 'unadressd'.strip('uned')
print(result)  # Salida: 'unadress'
