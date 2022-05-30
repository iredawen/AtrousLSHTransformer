list1=[]
list2=[]


# list1.append([
#     [1111] for i in range(3)]
# )
list2.extend([
    [1111] for i in range(3)]
)
# for j in range(4):

#     for i in range(4):
#         list1.append(i)
#     list2.append(list1)
list3=[]
list3.append([list2,2])
print(list3)
print(list2)