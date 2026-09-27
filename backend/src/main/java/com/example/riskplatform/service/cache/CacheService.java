package com.example.riskplatform.service.cache;import java.time.Duration;public interface CacheService{String get(String k);void set(String k,String v,Duration ttl);}
