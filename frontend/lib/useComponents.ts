'use client';

import { useCallback, useEffect, useState } from 'react';

import { ApiError, getComponents } from './api';
import type { ComponentsResponse } from './types';

export function useComponents() {
  const [data, setData] = useState<ComponentsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      setData(await getComponents());
      setError(null);
    } catch (e) {
      setData(null);
      setError(e instanceof ApiError ? e.message : 'Could not load component status.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  return { data, error, loading, refresh };
}
