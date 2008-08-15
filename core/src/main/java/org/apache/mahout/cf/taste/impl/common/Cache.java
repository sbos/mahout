/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.cf.taste.impl.common;

import org.apache.mahout.cf.taste.common.TasteException;

/**
 * <p>An efficient Map-like class which caches values for keys. Values are not "put" into a {@link Cache};
 * instead the caller supplies the instance with an implementation of {@link Retriever} which can load the
 * value for a given key.</p>
 *
 * <p>The cache does not support <code>null</code> values or keys.</p>
 *
 * <p>Thanks to Amila Jayasooriya for helping evaluate performance of the rewrite of this class, as part of a
 * Google Summer of Code 2007 project.</p>
 */
public final class Cache<K, V> implements Retriever<K, V> {

  private final FastMap<K, V> cache;
  private final Retriever<? super K, ? extends V> retriever;

  /**
   * <p>Creates a new cache based on the given {@link Retriever}.</p>
   *
   * @param retriever object which can retrieve values for keys
   */
  public Cache(Retriever<? super K, ? extends V> retriever) {
    this(retriever, FastMap.NO_MAX_SIZE);
  }

  /**
   * <p>Creates a new cache based on the given {@link Retriever} and with given maximum size.</p>
   *
   * @param retriever object which can retrieve values for keys
   * @param maxEntries maximum number of entries the cache will store before evicting some
   */
  public Cache(Retriever<? super K, ? extends V> retriever, int maxEntries) {
    if (retriever == null) {
      throw new IllegalArgumentException("retriever is null");
    }
    if (maxEntries < 1) {
      throw new IllegalArgumentException("maxEntries must be at least 1");
    }
    cache = new FastMap<K, V>(11, maxEntries);
    this.retriever = retriever;
  }

  /**
   * <p>Returns cached value for a key. If it does not exist, it is loaded using a {@link Retriever}.</p>
   *
   * @param key cache key
   * @return value for that key
   * @throws TasteException if an exception occurs while retrieving a new cached value
   */
  public V get(K key) throws TasteException {
    V value;
    synchronized (cache) {
      value = cache.get(key);
    }
    if (value == null) {
      return getAndCacheValue(key);
    }
    return value;
  }

  /**
   * <p>Uncaches any existing value for a given key.</p>
   *
   * @param key cache key
   */
  public void remove(K key) {
    synchronized (cache) {
      cache.remove(key);
    }
  }

  /**
   * <p>Clears the cache.</p>
   */
  public void clear() {
    synchronized (cache) {
      cache.clear();
    }
  }

  private V getAndCacheValue(K key) throws TasteException {
    V value = retriever.get(key);
    synchronized (cache) {
      cache.put(key, value);
    }
    return value;
  }

  @Override
  public String toString() {
    return "Cache[retriever:" + retriever + ']';
  }

}