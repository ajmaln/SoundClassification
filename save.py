name = raw_input("File Name: ")
file = open(name+'.json', 'w')
js = model.to_json()
file.write(js)
file.close()
model.save_weights(name+'.h5')
