from blendtorch import btb

def main():
    # Note, need to linger a bit in order to wait for unsent messages to be transmitted
    # before exiting blender.
    btargs, remainder = btb.parse_blendtorch_args()
    duplex = btb.DuplexChannel(btargs.btsockets['CTRL'], lingerms=5000)
    msgs = duplex.recv()
    duplex.send(dict(got=msgs))
    duplex.send('end')    
main()