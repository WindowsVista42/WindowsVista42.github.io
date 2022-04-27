+++
title = "Lizard Wizard"
date = 2021-07-01

[taxonomies]
tags = [ "engine", "rendering", "ecs" ]
+++

[Repository Link](https://github.com/WindowsVista42/Lizard-Wizard-Parberry-4210)  
A game created by me, Ethan Flute, and Solomon Weatherby for Ian Parberry's 4210 Intro to Game Programming I.

<!-- more -->

## What is it
The premise of Lizard Wizard was that you were a lizard who happened to be a wizard, and you would have to fight through dungeons to get to the next level and eventually beat the game.
At the end of each of these levels you would have some kind of boss that you would have to kill to proceed and the end of the game you would have a larger bigger badder boss to fight.

In short, it was set up to be a 3D roguelike, and we had roughly 3 months to implement everything.

## Rendering
The first problem that had to be tackled for this project was to build a prototype 3D renderer for visualization. Parberry's game engine used
[**DirectX12**](https://docs.microsoft.com/en-us/windows/win32/direct3d12/what-is-directx-12-) under the hood, and more specifically
[**DirectX12TK**](https://github.com/microsoft/DirectXTK12). To build a 3D renderer in his engine, I first had to extend the **LRenderer3D** class and add the necessary
stuff to get 3D rendering up and running. I won't go into the details of how DirectX12 works here, but for the prototype renderer I used DirectX12TK's
[**Primitive Batch**](https://github.com/microsoft/DirectXTK12/wiki/PrimitiveBatch). This allowed me and my team to push some vertex info to the GPU while I got to work on
a deferred renderer.

The rendering architecture I decided on for was a [**deferred renderer**](https://en.wikipedia.org/wiki/Deferred_shading). The benefit of this kind of approach
is that you get all of the vertex information after all of the vertices have been processed. Because of this, you don't have to worry about any data dependence
and you can process all rendering effects as post process effects. In the case of Lizard Wizard, this allowed me to create some simple lighting,
and my own take on [**ambient occlusion**](https://en.wikipedia.org/wiki/Ambient_occlusion). I chose this kind of architecture because our target platform was
high performance desktop PC's, and the high memory bandwidth usage isn't as much of a problem like it would be on mobile.

// PUT SOME IMPLEMENTATION DETAILS / CODE HERE

## Entity Component System
For this project I also built an [**entity component system**](https://en.wikipedia.org/wiki/Entity_component_system) as a way to learn how the technology works,
and to help us manage all of the stuff we would have in the scene.
The entity component system was broken down into three sections, entities, tables and groups.

An **Entity** in our system was a globally unique identifier for a thing in the scene, typically these would consist of 
A **Table** in our system was a list of components which, under the hood, was essentially an std::unordered_map mapping entities to components.
A **Group** in our system was where this all came together. Groups were a way to organize given entities into groups of similar functionality.

This was all combined with some helper functions like:
```
template <typename F>
Ecs::ApplyEvery(Group group, F f) {
  /* function details */
}
```
That would take a given group, a function, and apply that function to every member of the group.

One common way all of this was used was to iterate over the currently active entities of some type (denoted in a group), and apply some kind of update function to them.
```c++
// In Model.h
// Instance of a model
// We store the index of the model we want to render as,
// and we store the model matrix
struct ModelInstance {
    u32 model;      // model to render with
    u32 texture;    // texture to render with
    Vec3 glow;      // surface glow
    XMMATRIX world; // world matrix in world * view * projection
};

// In Renderer.h
class Renderer: public LRenderer3D {
    /* other fields */
    void DrawModelInstance(ModelInstance* instance);
    /* other fields */
};

// In Game.h
// Declaring the table and group
Table<ModelInstance> m_ModelInstances;
Group m_ModelsActive;

// In Game.cpp
// Draw the things
Ecs::ApplyEvery(m_ModelsActive, [&](Entity e) {
    m_pRenderer->DrawModelInstance(m_ModelInstances.Get(e));
});
```
The above usage would take everything that currently needs to be drawn and actually draw it.

## Procedural Generation
Procedural generation

## Movement

## Time Management

## Lessons Learned
