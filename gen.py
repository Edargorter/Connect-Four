f = open("td.txt", 'w')
count = 1
for i in range(5):
	for j in range(200):
		data = ",".join([str(count) for k in range(11)])
		f.write("{}\n".format(data))
	count += 1
f.close()
