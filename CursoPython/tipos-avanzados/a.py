if __name__ == '__main__':
    users = []
    for _ in range(int(input())):
        name = input()
        score = float(input())
        users.append([name, score])
    
    users.sort(key=lambda el: el[1])
    
    primero = users[0][1]
    segundo = None
    
    for user in users:
        if user[1] > primero:
            segundo = user[1]
            break
    
    userSort = [user[0] for user in users if user[1] == segundo]
    
    userSort.sort()
    for user in userSort:
        print(user)