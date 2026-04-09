import sys

output = ''


def _print(arg):
    global output
    output += arg


def _call(arg, stack):
    arg = int(arg)
    if arg < 1 or arg > len(program):
        print("ERROR")
        sys.exit(0)
    elif arg in stack:
        print("INFINITY")
        sys.exit(0)
    else:
        ip, cmd, arg = program[arg]
        stack.append(ip)
        if cmd == 'print':
            _print(arg)
        elif cmd == 'call':
            _call(arg, stack)


n = int(input())
program = ['*']
for ip in range(n):
    cmd, arg = input().split()
    program.append([ip+1, cmd, arg])

for ip, cmd, arg in program[1:]:
    if cmd == 'print':
        _print(arg)
    elif cmd == 'call':
        _call(arg, [ip])
else:
    print("OK")
    print(output)


# [ip1, cmd1, arg1] [ip2, cmd2, arg2] ... [ipN, cmdN, argN]

# [1, print, a] [2, call, 4] [3, call, 2] [4, print, b]
#
