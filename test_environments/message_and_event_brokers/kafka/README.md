# Testing Kafka broker

This is a set of various configurations under which Kafka message broker can be run.
Use them to spin up Kafka as a Docker container in a predefined configuration.

### Configuration description
Each configuration is described in its own README file.
Typical configuration includes:
* Kafka broker listening on port 9092
* Zookeeper listening on port 2181

#### About Zookeeper
Kafka and ZooKeeper work in conjunction to form a complete Kafka Cluster — with ZooKeeper 
providing the distributed clustering services, 
and Kafka handling the actual data streams and connectivity to clients.  

At a detailed level, ZooKeeper handles the leadership election of Kafka brokers and 
manages service discovery as well as cluster topology so each broker knows 
when brokers have entered or exited the cluster, when a broker dies and who the 
preferred leader node is for a given topic/partition pair. 

It also tracks when topics are created or deleted from the cluster and maintains a topic list. 
In general, ZooKeeper provides an in-sync view of the Kafka cluster.  

### Available configurations
* [Kafka without any authentication](no_authentication/README.md)
* [Kafka with SASL_PLAIN authentication without TLS encryption](sasl_plain/no_tls/README.md)
* [Kafka with SASL_PLAIN authentication and TLS encryption](sasl_plain/with_tls/README.md)
* [Kafka with SASL_SCRAM authentication without TLS encryption](sasl_scram/no_tls/README.md)
* [Kafka with SASL_SCRAM authentication and TLS encryption](sasl_scram/with_tls/README.md)
