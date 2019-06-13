
DIRECTORY = 'C:/Users/bergj/Documents/Geroge Mason/Courses/2019-Spring/GMU- CS 584/HW4/data/base/ModeD/'

with open('{}{}'.format(DIRECTORY, 'File68.txt')) as f:
    for line in f:
        line_split = line.split(' \t')

        for num in line_split:
            print(num)
