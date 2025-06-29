import { apiClient, ApiError } from '../api';
import { createBrowserClient } from '@/lib/supabase';

// Mocking Supabase client
const mockSupabase = {
  auth: {
    getSession: jest.fn(),
  },
};

jest.mock('@/lib/supabase', () => ({
  createBrowserClient: jest.fn(() => mockSupabase),
}));

// Mocking fetch
global.fetch = jest.fn();

describe('apiClient', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    // Reset the mock implementation for getSession for each test
    mockSupabase.auth.getSession.mockResolvedValue({
      data: {
        session: {
          access_token: 'test-token',
        },
      },
    });
  });

  it('should make a GET request successfully', async () => {
    (fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => ({ success: true }),
    });

    const response = await apiClient.get('/test');
    expect(response).toEqual({ success: true });
    expect(fetch).toHaveBeenCalledWith('http://localhost:8000/test', expect.any(Object));
  });

  it('should make a POST request successfully', async () => {
    (fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      status: 201,
      json: async () => ({ id: 1 }),
    });

    const response = await apiClient.post('/test', { data: 'test' });
    expect(response).toEqual({ id: 1 });
    expect(fetch).toHaveBeenCalledWith('http://localhost:8000/test', expect.objectContaining({
      method: 'POST',
      body: JSON.stringify({ data: 'test' }),
    }));
  });

  it('should handle ApiError on failed requests', async () => {
    (fetch as jest.Mock).mockResolvedValueOnce({
      ok: false,
      status: 404,
      statusText: 'Not Found',
      json: async () => ({ detail: 'Resource not found' }),
    });

    await expect(apiClient.get('/not-found')).rejects.toThrow(ApiError);
    await expect(apiClient.get('/not-found')).rejects.toThrow('API Error: 404 Not Found');
  });

  it('should attach auth token to requests', async () => {
    (fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => ({}),
    });

    await apiClient.get('/secure-resource');

    expect(fetch).toHaveBeenCalledWith(expect.any(String), expect.objectContaining({
      headers: expect.objectContaining({
        'Authorization': 'Bearer test-token',
      }),
    }));
  });
});
