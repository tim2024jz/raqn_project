from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ipv4, arp

class RAQNController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(RAQNController, self).__init__(*args, **kwargs)
        self.mac_to_port = {}
        self.logger.info("RAQN SDN Controller initialized with OpenFlow 1.3")

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        """Install table-miss flow entry."""
        datapath = ev.msg.datapath
        ofproto, parser = datapath.ofproto, datapath.ofproto_parser

        # Table-miss: send unmatched packets to controller
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER, ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, priority=0, match=match, actions=actions)
        self.logger.info("Installed table-miss flow on switch %016x", datapath.id)

    def add_flow(self, datapath, priority, match, actions, buffer_id=None, idle_timeout=0, hard_timeout=0):
        """Add a flow to the switch."""
        ofproto, parser = datapath.ofproto, datapath.ofproto_parser
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]

        if buffer_id and buffer_id != ofproto.OFP_NO_BUFFER:
            mod = parser.OFPFlowMod(
                datapath=datapath, buffer_id=buffer_id, priority=priority,
                idle_timeout=idle_timeout, hard_timeout=hard_timeout,
                match=match, instructions=inst)
        else:
            mod = parser.OFPFlowMod(
                datapath=datapath, priority=priority,
                idle_timeout=idle_timeout, hard_timeout=hard_timeout,
                match=match, instructions=inst)

        datapath.send_msg(mod)
        self.logger.debug("Flow added: dpid=%016x match=%s actions=%s", datapath.id, match, actions)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        """Handle incoming packets from the switch."""
        msg, datapath = ev.msg, ev.msg.datapath
        ofproto, parser = datapath.ofproto, datapath.ofproto_parser
        in_port = msg.match['in_port']
        dpid = datapath.id

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)
        dst, src = eth.dst, eth.src

        # Learn MAC-to-port mapping
        self.mac_to_port.setdefault(dpid, {})
        self.mac_to_port[dpid][src] = in_port

        self.logger.info("PacketIn: switch=%016x in_port=%s src=%s dst=%s", dpid, in_port, src, dst)
 
        if pkt.get_protocol(arp.arp):
            self.logger.debug("ARP packet received: ignoring special processing.")
             
       
        if dst in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst]
            self.logger.debug("Known destination %s -> output port %s", dst, out_port)
        else:
            out_port = ofproto.OFPP_FLOOD
            self.logger.debug("Unknown destination %s -> flooding", dst)

        actions = [parser.OFPActionOutput(out_port)]

       
        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_src=src, eth_dst=dst)
            self.add_flow(datapath, priority=1, match=match, actions=actions, idle_timeout=30, hard_timeout=60)

        # Send packet out
        data = msg.data if msg.buffer_id == ofproto.OFP_NO_BUFFER else None
        out = parser.OFPPacketOut(
            datapath=datapath, buffer_id=msg.buffer_id,
            in_port=in_port, actions=actions, data=data)
        datapath.send_msg(out)
