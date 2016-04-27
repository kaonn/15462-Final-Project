#include "software_renderer.h"

#include <cmath>
#include <vector>
#include <iostream>
#include <algorithm>
#include <random>

#include "triangulation.h"

using namespace std;

namespace CMU462 {

// Implements SoftwareRenderer //

void SoftwareRendererImp::draw_svg( SVG& svg ) {

  // set top level transformation
  transformation = canvas_to_screen;

  // draw all elements
  for ( size_t i = 0; i < svg.elements.size(); ++i ) {
    draw_element(svg,svg.elements[i]);
  }

  // resolve and send to render target
  resolve();

}

void SoftwareRendererImp::set_sample_rate( size_t sample_rate ) {

  // Task 3: 
  // You may want to modify this for supersampling support
  this->sample_rate = sample_rate;
  
  // set super_sample_target
  printf("sample rate changed!\n");

  set_render_target(render_target, target_w, target_h);
}

void SoftwareRendererImp::set_render_target( unsigned char* render_target,
                                             size_t width, size_t height ) {

  // Task 3: 
  // You may want to modify this for supersampling support
  this->render_target = render_target;
  this->target_w = width;
  this->target_h = height;

  printf("target_w: %lu target_h: %lu\n", width, height);
  super_w = sample_rate * (target_w);
  super_h = sample_rate * (target_h);
  supersample_buffer.resize(4*(super_w)*(super_h),0);
  histogram.resize((super_w)*(super_h),0);
  printf("target set!\n");
}

void SoftwareRendererImp::draw_element(SVG& svg, SVGElement* element) {

  // Task 4 (part 1):
  // Modify this to implement the transformation stack

  Matrix3x3 original = transformation;
  transformation = transformation * (element->transform);
  switch(element->type) {
    case POINT:
      draw_point(static_cast<Point&>(*element));
      break;
    case LINE:
      draw_line(static_cast<Line&>(*element));
      break;
    case POLYLINE:
      draw_polyline(static_cast<Polyline&>(*element));
      break;
    case RECT:
      draw_rect(static_cast<Rect&>(*element));
      break;
    case POLYGON:
      draw_polygon(static_cast<Polygon&>(*element));
      break;
    case ELLIPSE:
      draw_ellipse(static_cast<Ellipse&>(*element));
      break;
    case IMAGE:
      draw_image(static_cast<Image&>(*element));
      break;
    case IFS:
      draw_ifs(svg,static_cast<Ifs&>(*element));
      break;
    case GROUP:
      draw_group(svg,static_cast<Group&>(*element));
      break;
    default:
      break;
  }
  transformation = original;
}


// Primitive Drawing //

void SoftwareRendererImp::draw_point( Point& point ) {

  Vector2D p = transform(point.position);
  rasterize_point( p.x, p.y, point.style.fillColor );

}

void SoftwareRendererImp::draw_line( Line& line ) { 

  Vector2D p0 = transform(line.from);
  Vector2D p1 = transform(line.to);
  rasterize_line( p0.x, p0.y, p1.x, p1.y, line.style.strokeColor );

}

void SoftwareRendererImp::draw_polyline( Polyline& polyline ) {

  Color c = polyline.style.strokeColor;

  if( c.a != 0 ) {
    int nPoints = polyline.points.size();
    for( int i = 0; i < nPoints - 1; i++ ) {
      Vector2D p0 = transform(polyline.points[(i+0) % nPoints]);
      Vector2D p1 = transform(polyline.points[(i+1) % nPoints]);
      rasterize_line( p0.x, p0.y, p1.x, p1.y, c );
    }
  }
}

void SoftwareRendererImp::draw_rect( Rect& rect ) {

  Color c;
  
  // draw as two triangles
  float x = rect.position.x;
  float y = rect.position.y;
  float w = rect.dimension.x;
  float h = rect.dimension.y;

  Vector2D p0 = transform(Vector2D(   x   ,   y   ));
  Vector2D p1 = transform(Vector2D( x + w ,   y   ));
  Vector2D p2 = transform(Vector2D(   x   , y + h ));
  Vector2D p3 = transform(Vector2D( x + w , y + h ));
  
  // draw fill
  c = rect.style.fillColor;
  if (c.a != 0 ) {
    rasterize_triangle( p0.x, p0.y, p1.x, p1.y, p2.x, p2.y, c );
    rasterize_triangle( p2.x, p2.y, p1.x, p1.y, p3.x, p3.y, c );
  }

  // draw outline
  c = rect.style.strokeColor;
  if( c.a != 0 ) {
    rasterize_line( p0.x, p0.y, p1.x, p1.y, c );
    rasterize_line( p1.x, p1.y, p3.x, p3.y, c );
    rasterize_line( p3.x, p3.y, p2.x, p2.y, c );
    rasterize_line( p2.x, p2.y, p0.x, p0.y, c );
  }

}

void SoftwareRendererImp::draw_polygon( Polygon& polygon ) {

  Color c;

  // draw fill
  c = polygon.style.fillColor;
  if( c.a != 0 ) {

    // triangulate
    vector<Vector2D> triangles;
    triangulate( polygon, triangles );

    // draw as triangles
    for (size_t i = 0; i < triangles.size(); i += 3) {
      Vector2D p0 = transform(triangles[i + 0]);
      Vector2D p1 = transform(triangles[i + 1]);
      Vector2D p2 = transform(triangles[i + 2]);
      rasterize_triangle( p0.x, p0.y, p1.x, p1.y, p2.x, p2.y, c );
    }
  }

  // draw outline
  c = polygon.style.strokeColor;
  if( c.a != 0 ) {
    int nPoints = polygon.points.size();
    for( int i = 0; i < nPoints; i++ ) {
      Vector2D p0 = transform(polygon.points[(i+0) % nPoints]);
      Vector2D p1 = transform(polygon.points[(i+1) % nPoints]);
      rasterize_line( p0.x, p0.y, p1.x, p1.y, c );
    }
  }
}

void SoftwareRendererImp::draw_ellipse( Ellipse& ellipse ) {

  // Extra credit 

}
Color blend(Color element,vector<unsigned char> &buffer,size_t width, 
            int i, int j){
  float ea = element.a;
  float er = ea * element.r;
  float eg = ea * element.g;
  float eb = ea * element.b;
  float ca = (float)buffer[4*(i+j*width)+3]/255;
  float cr = (float)buffer[4*(i+j*width)]/255 * ca;
  float cg = (float)buffer[4*(i+j*width)+1]/255 * ca;
  float cb = (float)buffer[4*(i+j*width)+2]/255 * ca;

  float ca_p = 1 - (1-ea) * (1-ca);
  float cr_p = (1-ea)*cr + er;
  float cg_p = (1-ea)*cg + eg;
  float cb_p = (1-ea)*cb + eb;

  return Color(cr_p, cg_p, cb_p, ca_p);
}

void SoftwareRendererImp::update_histogram(double x, double y, Color color){
    size_t sx = (size_t) x % super_w;
    size_t sy = (size_t) y % super_h;

    if(histogram[sx + sy * super_w] == freq_max){
        freq_max++;
    }
    histogram[sx + sy * super_w]++;
    auto r = supersample_buffer[4*(sx + sy * super_w)];
    auto g = supersample_buffer[4*(sx + sy * super_w) + 1];
    auto b = supersample_buffer[4*(sx + sy * super_w) + 2];
    auto a = supersample_buffer[4*(sx + sy * super_w) + 3];
    Color c = blend(color,supersample_buffer,super_w,sx,sy);
    //supersample_buffer[4*(sx + sy * super_w)] = (uint8_t)(r/2 + 127.5*c.r);
    //supersample_buffer[4*(sx + sy * super_w)+1] = (uint8_t)(g/2 + 127.5*c.g);
    //supersample_buffer[4*(sx + sy * super_w)+2] = (uint8_t)(b/2 + 127.5*c.b);
    //supersample_buffer[4*(sx + sy * super_w)+3] = (uint8_t)(a/2 + 127.5*c.a);
      supersample_buffer[4 * (sx + sy * super_w)    ] = (uint8_t) (c.r * 255);
      supersample_buffer[4 * (sx + sy * super_w) + 1] = (uint8_t) (c.g * 255);
      supersample_buffer[4 * (sx + sy * super_w) + 2] = (uint8_t) (c.b * 255);
      supersample_buffer[4 * (sx + sy * super_w) + 3] = (uint8_t) (c.a * 255);
}

void SoftwareRendererImp::draw_ifs(SVG& svg, Ifs& ifs){
    double x = ((double)rand() / (double)RAND_MAX) ;
    double y = ((double)rand() / (double)RAND_MAX) ;
    double r = ((double)rand() / (double)RAND_MAX) ;
    double g = ((double)rand() / (double)RAND_MAX) ;
    double b = ((double)rand() / (double)RAND_MAX) ;
    double a = ((double)rand() / (double)RAND_MAX) ;
    
    printf("svg w: %f, svg h: %f\n",svg.width, svg.height);
    gamma = ifs.gamma;
    int num_fcts = ifs.system.size();
    int iters = ifs.num_iter;
    size_t fct_index;
    Matrix3x3 mat;
    Color c(r,g,b,a);

    isIFS = true;
    std::random_device rd;
    std::mt19937 gen(rd());

    std::discrete_distribution<> d(ifs.pdf.begin(),ifs.pdf.end());
    for(int i = 0 ; i < iters; i++){
        //fct_index = rand() % num_fcts;
        fct_index = d(gen);
        mat = ifs.system[fct_index];
        Vector3D mapped = mat * Vector3D(x,y,1);
        c = 0.5*(c + ifs.colors[fct_index]);
        x = sin(M_PI*(mapped.x)/2);
        y = sin(M_PI*(mapped.y)/2);
        double r2 = mapped.x * mapped.x + mapped.y*mapped.y;
        //x = mapped.x/r2;
        //y = mapped.y/r2;
        
        if(i < 20) continue;
        //rasterize_point((x+1)/2 * (double)svg.width + (double)(target_w - svg.width)/2, (y+1)/2 * (double)svg.height + (double)(target_h - svg.height)/2, Color(1,0,0,1));
        //update_histogram((x) * (double), (y) * (double)super_h, c);
        //update_histogram((x+1)/2 * (double)(sample_rate*svg.width) + (double)(super_w - sample_rate*svg.width)/2, (y+1)/2 * (double)(sample_rate*svg.height) + (double)(super_h - sample_rate*svg.height)/2, c);
        update_histogram((x+1)/2 * (double)(sample_rate*svg.width) + (double)(super_w - sample_rate*svg.width)/2, (y+1)/2 * (double)(sample_rate*svg.height) + (double)(super_h - sample_rate*svg.height)/2, c);
    }

}

void SoftwareRendererImp::draw_image( Image& image ) {

  Vector2D p0 = transform(image.position);
  Vector2D p1 = transform(image.position + image.dimension);

  rasterize_image( p0.x, p0.y, p1.x, p1.y, image.tex );
}

void SoftwareRendererImp::draw_group(SVG& svg, Group& group ) {

  for ( size_t i = 0; i < group.elements.size(); ++i ) {
    draw_element(svg,group.elements[i]);
  }

}

// Rasterization //

// The input arguments in the rasterization functions 
// below are all defined in screen space coordinates

void SoftwareRendererImp::rasterize_point( float x, float y, Color color ) {

  // fill in the nearest pixel
  float sx = (int) floor(x) + 0.5;
  float sy = (int) floor(y) + 0.5;

  // check bounds
  if ( sx < 0 || sx >= super_w) return;
  if ( sy < 0 || sy >= super_h) return;

  //sx = (int)(floor(sx * sample_rate));
  //sy = (int)(floor(sy * sample_rate));
  //Color c = blend(color,supersample_buffer,super_w,sx,sy);
  //supersample_buffer[4 * (sx + sy * super_w)    ] = (uint8_t) (c.r * 255);
  //supersample_buffer[4 * (sx + sy * super_w) + 1] = (uint8_t) (c.g * 255);
  //supersample_buffer[4 * (sx + sy * super_w) + 2] = (uint8_t) (c.b * 255);
  //supersample_buffer[4 * (sx + sy * super_w) + 3] = (uint8_t) (c.a * 255);
  // fill sample - NOT doing alpha blending!
  float sr = (float)sample_rate;
  int start_x = floor((sx - sr/(sr+1)*0.5)*sr);
  int start_y = floor((sy - sr/(sr+1)*0.5)*sr);
  int end_x = floor((sx + sr/(sr+1)*0.5)*sr);
  int end_y = floor((sy + sr/(sr+1)*0.5)*sr);
  Color c;
  for(int j = start_y; j <= end_y; j++)
    for(int i = start_x; i <= end_x; i++){
      c = blend(color,supersample_buffer,super_w,i,j);
      supersample_buffer[4 * (i + j * super_w)    ] = (uint8_t) (c.r * 255);
      supersample_buffer[4 * (i + j * super_w) + 1] = (uint8_t) (c.g * 255);
      supersample_buffer[4 * (i + j * super_w) + 2] = (uint8_t) (c.b * 255);
      supersample_buffer[4 * (i + j * super_w) + 3] = (uint8_t) (c.a * 255);
    }

}

void SoftwareRendererImp::rasterize_line( float x0, float y0,
                                          float x1, float y1,
                                          Color color) {

  // Task 1: 
  // Implement line rasterization
  float s,u,v;
  int sx, sy;

  if(x1 == x0 && y1 == y0){
    rasterize_point(x0,y0,color);
    return;
  }

  if(abs(x1-x0) > abs(y1-y0)){
    if(x1 < x0){
      float temp = x1;
      x1 = x0;
      x0 = temp;

      temp = y1;
      y1 = y0;
      y0 = temp;
    }

    s = (y1-y0)/(x1-x0);
    v = y0;

    for(float u = x0; u <= x1; u++){
      v += s;
      rasterize_point(u,v,color);
    }
  }
  else{
    if(y1 < y0){
      float temp = x1;
      x1 = x0;
      x0 = temp;

      temp = y1;
      y1 = y0;
      y0 = temp;
    }

    s = (x1-x0)/(y1-y0);
    u = x0;

    for(float v = y0; v <= y1; v++){
      u += s;
      rasterize_point(u,v,color);
    }
  }
}

int is_in(float x, float y, Vector3D l1, Vector3D l2, Vector3D l3){
  Vector3D p = Vector3D(x,y,1);
  float d1 = (dot(p,l1));
  float d2 = (dot(p,l2));
  float d3 = (dot(p,l3));
  return ((d1 < 0) &&  (d2 < 0) && (d3 < 0));
}

void SoftwareRendererImp::rasterize_triangle( float x0, float y0,
                                              float x1, float y1,
                                              float x2, float y2,
                                              Color color ) {
  // Task 2: 
  // Implement triangle rasterization


  Vector2D v01 = Vector2D(x1-x0, y1-y0);
  Vector2D v02 = Vector2D(x2-x0, y2-y0);

  if(cross(v01,v02) < 0){
    float temp = x1;
    x1 = x2;
    x2 = temp;

    temp = y1;
    y1 = y2;
    y2 = temp;
  }

  float dX0 = x1 - x0;
  float dY0 = y1 - y0;
  Vector3D l0 = Vector3D(dY0, -dX0, -x0 * dY0 + y0 * dX0);

  float dX1 = x2 - x1;
  float dY1 = y2 - y1;
  Vector3D l1 = Vector3D(dY1, -dX1, -x1 * dY1 + y1 * dX1);

  float dX2 = x0 - x2;
  float dY2 = y0 - y2;
  Vector3D l2 = Vector3D(dY2, -dX2, -x2 * dY2 + y2 * dX2);

  float u,v,y_max,y_min,x_max,x_min;
  y_min = min(y0,min(y1,y2));
  y_max = max(y0,max(y1,y2));
  x_max = max(x0,max(x1,x2));
  x_min = min(x0,min(x1,x2));
    
  float increment = 1.0/(sample_rate);
  float offset = increment/2;
  int sx,sy;
  Color c;
  for(v = floor(y_min) + offset; v <= ceil(y_max); v += increment){
    for(u = floor(x_min) + offset; u <= ceil(x_max); u += increment){
      if(is_in(u,v,l0,l1,l2)){
        sx = floor(u * sample_rate);
        sy = floor(v * sample_rate);
        if ( sx < 0 || sx >= super_w) return;
        if ( sy < 0 || sy >= super_h) return;
        c = blend(color,supersample_buffer,super_w,sx,sy);
        supersample_buffer[4 * (sx + sy * super_w)    ] = (uint8_t) (c.r*255); 
        supersample_buffer[4 * (sx + sy * super_w) + 1] = (uint8_t) (c.g*255); 
        supersample_buffer[4 * (sx + sy * super_w) + 2] = (uint8_t) (c.b*255); 
        supersample_buffer[4 * (sx + sy * super_w) + 3] = (uint8_t) (c.a*255); 
      }
    }
  }

}

void SoftwareRendererImp::rasterize_image( float x0, float y0,
                                           float x1, float y1,
                                           Texture& tex ) {
  // Task ?: 
  // Implement image rasterization
  //

  

  //Bilinear
  //Color c; 
  //Sampler2DImp s = Sampler2DImp(BILINEAR);
  //float u,v;
  //for(float j = y0; j <= y1; j++){
      //for(float i = x0 ; i <= x1; i++){
        //u = (i-x0)/(x1-x0);
        //v = (j-y0)/(y1-y0);
        //c = s.sample_bilinear(tex,u,v,1);
        ////printf("u: %f v: %f\n", u, v);
        //rasterize_point(i,j,c);
      //}
  //}
  

  //Trilinear

  size_t text_width = (tex.mipmap)[0].width;
  size_t text_height = (tex.mipmap)[0].height;
  float u_scale = (float)text_width/(x1-x0);
  float v_scale = (float)text_height/(y1-y0);

  Color c; 
  Sampler2DImp s = Sampler2DImp(TRILINEAR);
  float u,v,u1,v1;
  for(float j = floor(y0) + 0.5; j <= ceil(y1); j++){
      for(float i = floor(x0) + 0.5; i <= ceil(x1); i++){
        u = (i-x0)/(x1-x0);
        v = (j-y0)/(y1-y0);
        c = s.sample_trilinear(tex,u,v,u_scale,v_scale);
        rasterize_point(i,j,c);
      }
  }


}

// resolve samples to render target
void SoftwareRendererImp::resolve( void ) {

  // Task 3: 
  // Implement supersampling
  // You may also need to modify other functions marked with "Task 3".
  //
  size_t blocksize = sample_rate;
  float r=0.0,g=0.0,b=0.0,a=0.0,alpha = 0.0;
  unsigned int total_freq = 0;
  for(size_t j = 0; j < target_h; j++){
    for(size_t i = 0; i < target_w; i++){

      r = 0.0;g=0.0;b=0.0,a=0.0,total_freq = 0;
      
      for(size_t l = blocksize*j; l < blocksize*j + blocksize; l++){
        for(size_t k = blocksize*i; k < blocksize*i + blocksize; k++){
           r += supersample_buffer[4*(k + l*super_w)];
           g += supersample_buffer[4*(k + l*super_w) + 1];
           b += supersample_buffer[4*(k + l*super_w) + 2];
           a += supersample_buffer[4*(k + l*super_w) + 3];

           total_freq += histogram[(k + l*super_w)];
        }
      }
      a = log(total_freq)/log(freq_max);
      alpha = pow(a,(double)1/gamma);

      render_target[4 * (i + j * target_w)    ] = (uint8_t) (r/(blocksize*blocksize) ); 
      render_target[4 * (i + j * target_w) + 1] = (uint8_t) (g/(blocksize*blocksize) ); 
      render_target[4 * (i + j * target_w) + 2] = (uint8_t) (b/(blocksize*blocksize) ); 
      render_target[4 * (i + j * target_w) + 3] = 255*alpha; 
      //if(total_freq < 10)
          //render_target[4 * (i + j * target_w) + 3] = 255; 

    }
  }
  printf("finished resolving!\n");
  printf("target_w: %lu target_h: %lu\n", target_w, target_h);
  printf("max: %lu\n", freq_max);
  clear_supersample_buffer();
  return;

}


} // namespace CMU462
