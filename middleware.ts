import { createServerClient, type CookieOptions } from '@supabase/ssr';
import { type NextRequest, NextResponse } from 'next/server';

const publicPaths = [
  '/login',
  '/register',
  '/auth/callback',
];

export async function middleware(request: NextRequest) {
  const path = request.nextUrl.pathname;

  // Create an outgoing response object that can be modified
  // Clone request headers to ensure they are passed through
  let response = NextResponse.next({
    request: {
      headers: request.headers,
    },
  });

  // Check if the current path is one of the public paths
  // or if it's for Next.js internals that should always be allowed.
  if (publicPaths.includes(path) || path.startsWith('/api/') || path.startsWith('/_next/')) {
    return response; // Use the prepared response
  }

  const supabase = createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        get(name: string) {
          return request.cookies.get(name)?.value;
        },
        set(name: string, value: string, options: CookieOptions) {
          // If the cookie is set, update the request and response cookies
          request.cookies.set({ name, value, ...options });
          response.cookies.set({ name, value, ...options });
        },
        remove(name: string, options: CookieOptions) {
          // If the cookie is removed, update the request and response cookies
          request.cookies.set({ name, value: '', ...options });
          response.cookies.set({ name, value: '', ...options });
        },
      },
    }
  );

  try {
    const { data: { session }, error: sessionError } = await supabase.auth.getSession();

    if (sessionError) {
      console.error('Error getting session in middleware:', sessionError.message);
      // Potentially redirect to an error page or login
      const redirectUrl = request.nextUrl.clone();
      redirectUrl.pathname = '/login';
      redirectUrl.search = '';
      return NextResponse.redirect(redirectUrl);
    }

    if (!session) {
      const redirectUrl = request.nextUrl.clone();
      redirectUrl.pathname = '/login';
      redirectUrl.search = ''; // Clear search params
      return NextResponse.redirect(redirectUrl);
    }

    // Session exists, allow the request to proceed with the (potentially modified) response
    return response;

  } catch (error) {
    console.error('Unexpected middleware error on path:', path, error);
    const errorRedirectUrl = request.nextUrl.clone();
    errorRedirectUrl.pathname = '/login'; // Or a generic error page
    errorRedirectUrl.search = '';
    return NextResponse.redirect(errorRedirectUrl);
  }
}

export const config = {
  matcher: [
    /*
     * Match all request paths except for the ones starting with:
     * - _next/static (static files)
     * - _next/image (image optimization files)
     * - favicon.ico (favicon file)
     * - public (directory) - This excludes all files in the /public folder
     */
    '/((?!_next/static|_next/image|favicon.ico|public/).*)',
  ],
};
