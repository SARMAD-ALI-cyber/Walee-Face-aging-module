import requests

headers = {
    'accept': 'application/json',
    'x-magicapi-key': 'cm68rmypo0001l803j2roq9cu',
    'Content-Type': 'application/json',
}

data = '{\n  "image": https://www.dropbox.com/scl/fi/l1uyy1kh0vcv1u4d04rjl/nosh.png?rlkey=x70wdrisx5m6qfdlisacsxvb3&st=i4c7qrva&dl=1,\n  "target_age": "90"\n}'

response = requests.post('https://api.magicapi.dev/api/v1/magicapi/period/period', headers=headers, data=data)
print(response)



headers = {
    'accept': 'application/json',
    'x-magicapi-key': 'cm68qhkwk0002l203eb10zt28',
}

response = requests.get(f'https://api.magicapi.dev/api/v1/magicapi/period/predictions/{response["request_id"]}', headers=headers)
print(response)