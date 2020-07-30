from blendtorch import btb

def main():
    # Note, need to linger a bit in order to wait for unsent messages to be transmitted
    # before exiting blender.
    btargs, remainder = btb.parse_blendtorch_args()
    pub = btb.DataPublisher(btargs.btsockets['DATA'], btargs.btid, lingerms=10000)
    pub.publish(btargs=vars(btargs), remainder=remainder)
main()