from blendtorch import btb

def main():
    # Note, need to linger a bit in order to wait for unsent messages to be transmitted
    # before exiting blender.
    btargs, remainder = btb.parse_blendtorch_args()
    duplex = btb.DuplexChannel(btargs.btsockets['CTRL'], btid=btargs.btid, lingerms=5000)
    
    msg = duplex.recv(timeoutms=5000)
    duplex.send(echo=msg)
    duplex.send(msg='end') 
main()