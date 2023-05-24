from global_land_mask import globe
c=0
v=0
w=0
for i in range(int((62.3 - 59.5) / 0.0003) + 1):
    c=c+1
    try:
        x = 59.5 + i * 0.0003
        y = (24.6 - 17.6) / (59.5 - 62.3) * (x - 59.5) + 17.6
        print(x,y)
    except:
        continue
    is_on_land = globe.is_land(float(x), float(y))
    if not is_on_land:
        # print("in water")
        w=w+1
    if is_on_land:
        #print("crosses land")
        v=v+1
print("for loop run",c)
print("crosses land",v)
print("in water",w)
#print(crosses_land(62.3, 17.6, 59.5,24.6))
#crosses_land(59.5, 17.6, 62.3,24.6)
#
# for i in range(int((5 - 4) / 0.0003) + 1):
#     print(i)
#     print("hello")